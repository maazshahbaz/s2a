"""
LLM-Based Speaker Label Corrector using Mistral-Nemo via Triton.

Combined functionality:
1. IVR Detection - Identifies automated system messages
2. Speaker Role Assignment - Labels speakers as Agent/Customer/etc.
"""

import tritonclient.grpc.aio as grpcclient_aio
import numpy as np
import json
import uuid
import re
from typing import List, Dict, Tuple, Optional
import time


class LLMSpeakerCorrector:
    """
    Corrects diarization speaker labels using a combination of rule-based heuristics
    and an LLM (Mistral-Nemo) served via NVIDIA Triton Inference Server.

    The corrector performs two main tasks in a single LLM pass:
        1. **IVR Detection**: Identifies automated system messages at the start of a
           call (e.g., "Thank you for calling. Your call may be recorded.") and
           relabels them as 'IVR' instead of a generic speaker ID.
        2. **Role Assignment**: Classifies each human speaker as either 'Agent'
           (the outbound/calling party) or 'Customer' (the inbound/receiving party).

    To minimize latency, only a head+tail sample of the transcript is sent to the
    LLM. Rule-based pre- and post-processing guards handle common edge cases before
    and after the LLM call.

    Attributes:
        triton_url (str): gRPC address of the Triton Inference Server.
        model_name (str): Name of the model registered in Triton.
        client: Async Triton gRPC client instance.
    """

    def __init__(self, triton_url: str = None, model_name: str = None):
        """
        Initialises the corrector and immediately creates a Triton gRPC client.

        Args:
            triton_url (str, optional): Host:port of the Triton server.
                Defaults to 'localhost:3701'.
            model_name (str, optional): Model name as registered in Triton.
                Defaults to 'mistral-nemo'.

        Note:
            The client is created eagerly here (not lazily) so that connection
            errors surface at construction time rather than at first inference.
        """
        self.triton_url = triton_url or 'localhost:3701'
        self.model_name = model_name or 'mistral-nemo'
        self.client = None
        # Eagerly initialise the async gRPC client so callers get early feedback
        # if the Triton server is unreachable.
        self.client = grpcclient_aio.InferenceServerClient(url=self.triton_url)

    def _get_system_prompt(self) -> str:
        """
        Returns the system-level instruction prompt sent to the LLM.

        Returns:
            str: The full system prompt string.
        """
        return """You are an expert at analyzing customer service call transcripts.

## TASK 1: IVR DETECTION
IVR (Interactive Voice Response) is an AUTOMATED system message that appears ONLY at the very START of a call.
IVR examples:
- "Thank you for calling [Company]. Your call may be recorded."
- "Please wait while we connect you to an agent."
- "Press 1 for sales, press 2 for support."
- "Thank you for calling MTC, home of the Fixture-Free Guarantee. Please wait while we answer your call."
NOT IVR:
- Human greetings like "Hello", "Hi, how are you?"
- Agent introductions like "Hi, my name is Ryan"
- Any back-and-forth conversation
Key rule: IVR is ONE-WAY automated announcements. Once a human responds or conversation begins, IVR is over.

## TASK 2: SPEAKER ROLES
Label human speakers as Agent or Customer.
The Agent is the person who INITIATED the call or represents the calling company.
The Customer is the person who RECEIVED the call or works at the company being called.

CRITICAL IDENTIFICATION RULES (in PRIORITY ORDER):
### HIGHEST PRIORITY — these ALWAYS identify the AGENT:
1. Says "I'm with [Company]" or "I'm from [Company]" or "This is [Name] from [Company]"
2. Says "I'm calling from [Company]" or "I'm calling you from [Company]"
3. Says "I'm looking to speak with [Name]" or "I'm trying to reach [Name]"
4. Says "I'm reaching out to you about..." or "I'm calling regarding..."
5. Explains their PURPOSE for calling
### HIGHEST PRIORITY — these ALWAYS identify the CUSTOMER:
1. Answers the phone with a department name: "Building Office", "Billing Office", "Front Desk"
2. Says "I can take your information and forward it to my manager"
3. Says "What is this regarding?" or "Is he helping you with something?"
4. Takes a message: "I'll forward this to...", "I'll give this to my manager"

## RESPONSE FORMAT
Return ONLY a JSON object:
{
    "has_ivr": true/false,
    "ivr_text": "exact IVR text from transcript or empty string",
    "speaker_roles": {"speaker_0": "Agent" or "Customer", "speaker_1": "Agent" or "Customer"},
    "reasoning": "brief explanation of why you assigned each role"
}"""

    async def initialize(self):
        """
        Re-initialises the Triton gRPC client.

        This method is provided as a convenience hook for callers that manage
        the client lifecycle explicitly (e.g., after calling ``close``).  The
        constructor already creates a client, so calling this method is only
        necessary if the connection must be re-established.
        """
        self.client = grpcclient_aio.InferenceServerClient(url=self.triton_url)

    def _format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """
        Formats a system + user prompt pair into the Mistral instruction template.

        Mistral-Nemo uses the ``[INST] … [/INST]`` format to delimit instructions.
        Both the system prompt and user prompt are concatenated inside a single
        ``[INST]`` block, which is the recommended approach for Mistral models
        that do not natively support a separate ``<system>`` role token.

        Args:
            system_prompt (str): High-level behavioural instructions for the model.
            user_prompt (str): The per-request task description and transcript sample.

        Returns:
            str: The fully formatted prompt string ready for tokenisation.
        """
        return f"[INST] {system_prompt}\n\n{user_prompt} [/INST]"

    async def _generate_async(self, prompt: str, request_id: str = None) -> str:
        """
        Sends a prompt to the Triton Inference Server and returns the generated text.

        The prompt is packaged as a 1×1 BYTES tensor named ``"prompt"`` (the input
        binding expected by the Triton model configuration).  The server returns a
        corresponding ``"generated_text"`` tensor.

        Args:
            prompt (str): The fully formatted prompt string.
            request_id (str, optional): Unique identifier for this inference request,
                used for tracing/logging in Triton.  A UUID is generated if not supplied.

        Returns:
            str: The raw text generated by the model.

        Raises:
            Exception: Re-raises any exception from the Triton gRPC client after
                logging it, so callers can handle or propagate as needed.
        """
        if request_id is None:
            request_id = str(uuid.uuid4())
        try:
            # Triton requires numpy arrays as input; shape [1, 1] for a single string
            input_data = np.array([[prompt]], dtype=object)
            inputs = [grpcclient_aio.InferInput("prompt", [1, 1], "BYTES")]
            inputs[0].set_data_from_numpy(input_data)

            # Request only the generated text output binding
            outputs = [grpcclient_aio.InferRequestedOutput("generated_text")]

            response = await self.client.infer(
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs,
                request_id=request_id
            )
            # Decode bytes tensor to Python string
            output_text = response.as_numpy("generated_text")[0].decode('utf-8')
            return output_text
        except Exception as e:
            print(f"[LLM Corrector] Error during Triton inference: {e}")
            raise

    def _extract_json_from_response(self, response: str) -> Optional[Dict]:
        """
        Parses the first valid JSON object from a raw LLM response string.

        The LLM sometimes wraps its JSON in markdown code fences (```json … ```).
        This method first attempts to extract JSON from such a fence, then falls
        back to scanning the raw string for a balanced ``{ … }`` block.

        Args:
            response (str): The raw text output from the LLM.

        Returns:
            Optional[Dict]: The parsed JSON as a Python dict, or ``None`` if no
                valid JSON object could be found or parsed.
        """
        # Attempt 1: extract from ```json … ``` markdown code fence
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Attempt 2: find the first balanced { … } block in raw text
        start = response.find('{')
        if start != -1:
            depth = 0
            for i, char in enumerate(response[start:], start):
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(response[start:i+1])
                        except json.JSONDecodeError:
                            pass
                        break
        return None

    def _detect_speakers_in_transcript(self, labeled_transcription: str) -> List[str]:
        """
        Extracts an ordered, deduplicated list of speaker IDs from a transcript.

        Speaker IDs are expected to appear in the format ``[speaker_N]`` at the
        start of each transcript line (e.g., ``[speaker_0] Hello there``).

        Args:
            labeled_transcription (str): Multi-line transcript with ``[speaker_N]``
                prefixes on each utterance line.

        Returns:
            List[str]: Speaker IDs in the order they first appear in the transcript
                (e.g., ``['speaker_0', 'speaker_1']``).
        """
        speaker_pattern = re.compile(r'\[(speaker_\d+)\]')
        matches = speaker_pattern.findall(labeled_transcription)

        # Preserve first-appearance order while deduplicating
        seen = set()
        speakers = []
        for match in matches:
            if match not in seen:
                seen.add(match)
                speakers.append(match)
        return speakers

    def _is_ivr_like_text(self, text: str) -> bool:
        """
        Determines whether a text string contains IVR (automated system) language.

        Uses two tiers of keyword/phrase matching:
            - **Strong phrases** (e.g., "press 1", "your call may be recorded"):
              A single match is sufficient to flag the text as IVR.
            - **Weak phrases** (e.g., "please hold", "welcome to"):
              At least two matches are required, reducing false positives on
              human utterances that happen to use one of these phrases.

        This method is intentionally conservative to avoid misclassifying
        human speech as IVR.

        Args:
            text (str): The transcript text to evaluate (case-insensitive).

        Returns:
            bool: ``True`` if the text is likely IVR content, ``False`` otherwise.
        """
        text_lower = text.lower()

        # High-confidence IVR phrases — one match is conclusive
        strong_ivr_phrases = [
            "your call may be recorded", "this call may be monitored",
            "this call is being recorded", "for quality and training",
            "for quality assurance", "press 1", "press 2", "press star", "press pound",
            "say or press", "please listen carefully", "our menu options have changed",
            "your call is important to us", "your call will be answered",
            "estimated wait time", "you are caller number", "para espanol",
            "our office hours are", "we are currently closed",
            "leave a message after the tone", "no one is available to take your call",
            "your call is being transferred", "fixture-free guarantee",
            "at the tone, please record", "record your message",
            "when you have finished recording", "when you are finished recording",
            "your call has been forwarded", "call has been forwarded to voicemail",
            "the person you're trying to reach", "not available to take your call",
        ]

        # Lower-confidence IVR phrases — require at least two matches
        weak_ivr_phrases = [
            "welcome to", "please hold", "please wait while",
            "connecting you", "transferring your call", "for english",
            "please hold while", "home of the",
        ]

        strong_matches = sum(1 for p in strong_ivr_phrases if p in text_lower)
        weak_matches = sum(1 for p in weak_ivr_phrases if p in text_lower)
        return strong_matches >= 1 or weak_matches >= 2

    def _is_conversational_start(self, transcript: str) -> bool:
        """
        Heuristically checks whether a transcript begins with human conversation
        rather than an IVR announcement.

        Looks at the first line of the transcript for common conversational openers
        (e.g., "hello", "hi,", "good morning") and verifies they do not also match
        IVR patterns.  This guards against incorrectly flagging calls that start
        directly with a human greeting as having IVR.

        Args:
            transcript (str): The (possibly sampled) transcript text.

        Returns:
            bool: ``True`` if the transcript appears to start with human speech.
        """
        lines = transcript.strip().split('\n')
        if not lines:
            return True

        first_line = lines[0].lower()
        conversational_starters = ["hello.", "hi,", "hi.", "good afternoon", "good morning", "good evening"]
        for starter in conversational_starters:
            # Must match a conversational opener AND not contain IVR language
            if starter in first_line and not self._is_ivr_like_text(first_line):
                return True
        return False

    def _detect_outbound_agent(
        self,
        speaker_texts: Dict[str, List[str]],
        detected_speakers: List[str]
    ) -> Optional[str]:
        """
        Uses regex heuristics to identify the outbound-calling agent before the LLM call.

        Scans the first ~60 words of each speaker's turns for language that strongly
        implies the speaker initiated the call (e.g., "I'm calling from", "I received
        a voicemail", "I'm following up").  This override is applied after the LLM
        result to correct cases where the LLM misassigns the agent role.

        Two confidence tiers are used:
            - **HIGH patterns** (e.g., "this is X calling", "got your voicemail"):
              A single match forces the speaker to Agent.
            - **MEDIUM patterns** (e.g., "I'm reaching out", "I'm trying to reach"):
              Two or more matches are required.

        Args:
            speaker_texts (Dict[str, List[str]]): Mapping of speaker ID to a list of
                their utterance strings.  Only *human* (non-IVR) speakers should be
                passed in.
            detected_speakers (List[str]): Ordered list of speaker IDs to evaluate.

        Returns:
            Optional[str]: The speaker ID that should be forced to 'Agent', or
                ``None`` if no outbound-agent signal was detected.
        """
        # High-confidence outbound patterns — one match is sufficient
        OUTBOUND_HIGH = [
            r'\bthis is \w+ (again )?calling\b',
            r'\b(again )?calling (you )?back\b',
            r'\bcalling you from\b',
            r"\bi'?m calling from\b",
            r'\bcalling (regarding|about|concerning|in regards)\b',
            r'\bi received (a )?voicemail\b',
            r'\bgot (your )?voicemail\b',
            r'\bi (left|sent) (you )?(a )?voicemail\b',
            r'\bleft (you )?(a )?message\b',
            r"\bi'?m following up\b",
            r'\bfollowing up on (my |the )?(voicemail|message|call)\b',
            r'\bvoicemail this morning\b',
            r'\bvoicemail (yesterday|earlier|last week)\b',
        ]

        # Medium-confidence outbound patterns — require two or more matches
        OUTBOUND_MEDIUM = [
            r"\bi'?m reaching out\b",
            r"\bi'?m trying to reach\b",
            r"\bi'?m looking to speak with\b",
            r'\breason for my call\b',
            r'\bpurpose of my call\b',
            r"\bi'?m (calling|getting in touch) (to|because)\b",
        ]

        for spk in detected_speakers:
            words = speaker_texts.get(spk, [])
            if not words:
                continue
            # Only inspect the first ~60 words to mimic "opening statement" analysis
            opening = ' '.join(' '.join(words).split()[:60]).lower()

            high_hits = sum(1 for p in OUTBOUND_HIGH if re.search(p, opening))
            med_hits  = sum(1 for p in OUTBOUND_MEDIUM if re.search(p, opening))

            if high_hits >= 1 or med_hits >= 2:
                print(f"[Outbound] Agent: '{spk}' (high={high_hits} med={med_hits}) "
                      f"\"{opening[:80]}\"")
                return spk

        return None

    def _pre_classify_ivr_speakers(
        self,
        detected_speakers: List[str],
        labeled_transcription: str
    ) -> Dict[str, bool]:
        """
        Rule-based pre-pass to identify IVR speakers before calling the LLM.

        Iterates through all utterances for each speaker and checks whether the
        speaker's *entire* contribution contains IVR-like language AND is short
        enough (≤ 120 words) to plausibly be an automated announcement.  This
        avoids an unnecessary (and expensive) LLM round-trip for calls where the
        IVR can be identified deterministically.

        Args:
            detected_speakers (List[str]): Ordered list of all speaker IDs found in
                the transcript.
            labeled_transcription (str): Full multi-line transcript string.

        Returns:
            Dict[str, bool]: Mapping of speaker ID → ``True`` if classified as IVR,
                ``False`` otherwise.
        """
        # Collect all utterance text per speaker
        speaker_text: Dict[str, List[str]] = {spk: [] for spk in detected_speakers}
        for line in labeled_transcription.strip().split('\n'):
            m = re.match(r'\[([^\]]+)\]\s*(.*)', line)
            if m and m.group(1) in speaker_text:
                speaker_text[m.group(1)].append(m.group(2).strip().lower())

        is_ivr: Dict[str, bool] = {}
        for spk in detected_speakers:
            words = speaker_text.get(spk, [])
            full_text  = ' '.join(words)
            word_count = sum(len(w.split()) for w in words)

            # Classify as IVR only if word count is low AND phrases match
            if word_count <= 120 and self._is_ivr_like_text(full_text):
                is_ivr[spk] = True
                print(f"[Pre-IVR] '{spk}' classified as IVR ({word_count}w): '{full_text[:60]}...'")
            else:
                is_ivr[spk] = False
        return is_ivr

    async def analyze_transcript(
        self,
        labeled_transcription: str,
        request_id: str,
        aligned_words: Optional[List[Dict]] = None,
        sample_size: int = 30
    ) -> Tuple[str, Optional[List[Dict]], Dict]:
        """
        Main entry point: analyses a labeled transcript and returns corrected labels.

        Orchestrates the full correction pipeline:
            1. Detect speaker IDs in the transcript.
            2. Pre-classify IVR speakers using rule-based heuristics.
            3. Detect the outbound agent using regex patterns.
            4. Build a head+tail sample (first N/2 + last N/2 lines) to reduce
               token usage while preserving call context.
            5. Send the sample to the LLM for IVR detection and role assignment.
            6. Override / validate the LLM result with deterministic rules.
            7. Apply the final label map to both the transcript text and the
               word-level alignment data.

        Args:
            labeled_transcription (str): Multi-line transcript where each line is
                prefixed with ``[speaker_N]``.
            request_id (str): Unique request identifier passed to Triton for tracing.
            aligned_words (Optional[List[Dict]]): Word-level alignment data from the
                ASR pipeline.  Each dict is expected to contain at least ``"text"``
                and ``"speaker"`` keys.  If provided, speaker labels in this list are
                also corrected.
            sample_size (int): Maximum number of transcript lines to send to the LLM.
                Lines are taken as the first ``sample_size//2`` and last
                ``sample_size//2`` of the full transcript.  Defaults to 30.

        Returns:
            Tuple containing:
                - **corrected_transcript** (str): Transcript with updated speaker labels.
                - **corrected_words** (Optional[List[Dict]]): Word-level data with
                  updated speaker labels, or ``None`` if no alignment was provided.
                - **result** (Dict): The raw analysis result dict from the LLM
                  (after validation overrides), containing keys ``has_ivr``,
                  ``ivr_text``, ``speaker_roles``, and ``reasoning``.
        """
        t1 = time.time()

        # ── Step 1: Discover all speaker IDs ─────────────────────────────────
        detected_speakers = self._detect_speakers_in_transcript(labeled_transcription)

        # Fallback: if no [speaker_N] tags found, extract IDs from aligned words
        if not detected_speakers and aligned_words:
            speaker_set = set()
            for word in aligned_words:
                spk = word.get('speaker', '')
                if spk and not spk.startswith('IVR'):
                    speaker_set.add(spk)
            detected_speakers = sorted(list(speaker_set))

        if not detected_speakers:
            print("[LLM Corrector] No speakers detected")
            return labeled_transcription, aligned_words, {}

        print(f"[LLM Corrector] Detected {len(detected_speakers)} speakers: {detected_speakers}")

        # ── Step 2: Pre-classify IVR speakers (rule-based, no LLM cost) ──────
        pre_ivr_map = self._pre_classify_ivr_speakers(detected_speakers, labeled_transcription)
        ivr_pre_speakers   = [spk for spk, v in pre_ivr_map.items() if v]
        human_speakers_pre = [spk for spk in detected_speakers if not pre_ivr_map.get(spk)]

        if ivr_pre_speakers:
            print(f"[Pre-IVR] IVR: {ivr_pre_speakers}  Human: {human_speakers_pre}")

        # ── Step 3: Build per-speaker text maps for heuristic analysis ────────
        speaker_text_map: Dict[str, List[str]] = {spk: [] for spk in detected_speakers}
        for line in labeled_transcription.strip().split('\n'):
            m = re.match(r'\[([^\]]+)\]\s*(.*)', line)
            if m and m.group(1) in speaker_text_map:
                speaker_text_map[m.group(1)].append(m.group(2).strip())

        # Only pass human (non-IVR) speakers to the outbound-agent detector
        human_text_map = {
            spk: speaker_text_map[spk]
            for spk in human_speakers_pre if spk in speaker_text_map
        }
        outbound_agent_spk = self._detect_outbound_agent(human_text_map, human_speakers_pre)

        # ── Step 4: Build head+tail sample to cap LLM token usage ────────────
        lines = labeled_transcription.split('\n')
        half = sample_size // 2
        if len(lines) > sample_size:
            # Take equal slices from the start and end; middle is elided
            sample_lines = (
                lines[:half]
                + ['... [middle of call] ...']
                + lines[-half:]
            )
            sample_transcript = '\n'.join(sample_lines)
            print(f"[LLM] head+tail sample: first {half} + last {half} of {len(lines)} lines")
        else:
            sample_transcript = labeled_transcription

        starts_conversational = self._is_conversational_start(sample_transcript)

        # Build the dynamic speaker_roles format string for the user prompt
        speaker_roles_format = ', '.join(
            [f'"{spk}": "Agent" or "Customer"' for spk in detected_speakers]
        )

        # ── Step 5: Construct and send prompt to LLM ─────────────────────────
        user_prompt = f"""Analyze this transcript:

TRANSCRIPT:
{sample_transcript}

IMPORTANT - IDENTIFICATION PRIORITY:
1. HIGHEST PRIORITY: If someone says "I'm with [Company]", "I'm from [Company]", "calling from [Company]", or "I'm looking to speak with [Name]" -> they are ALWAYS the AGENT
2. HIGHEST PRIORITY: If someone answers with a department name ("Building Office", "Billing Office") or says "I'll forward this to my manager" -> they are ALWAYS the CUSTOMER
3. A receptionist asking "What's your name?" or "What is this about?" is a CUSTOMER doing screening, NOT an Agent asking diagnostic questions
4. In inbound support calls: the person who says "Thanks for calling, how can I help?" is the AGENT
5. The person describing their problem ("the monitor isn't working") is the CUSTOMER
6. VOICEMAIL / IVR: If the call goes to voicemail, the automated greeting ("Your call has been forwarded...") is IVR, NOT Customer. The person leaving the message is the AGENT.

Return JSON with:
- has_ivr: true if there is automated IVR at the START (false if it starts with human conversation)
- ivr_text: the exact IVR text (empty string if no IVR)
- speaker_roles: {{{speaker_roles_format}}}
- reasoning: brief explanation"""

        full_prompt = self._format_prompt(self._get_system_prompt(), user_prompt)

        print("[LLM] Analyzing transcript for roles + IVR...")
        response = await self._generate_async(full_prompt, request_id)
        result   = self._extract_json_from_response(response)

        # ── Step 6: Validate and override LLM result ─────────────────────────
        if result is None:
            print("[LLM] Warning: Could not parse LLM response")
            result = self._get_fallback_result(detected_speakers)
        else:
            # Force any pre-classified IVR speakers to IVR regardless of LLM output
            for spk in ivr_pre_speakers:
                if result.get('speaker_roles', {}).get(spk) in ('Agent', 'Customer'):
                    print(f"[Pre-IVR] Overriding '{spk}': {result['speaker_roles'][spk]} -> IVR")
                    result['speaker_roles'][spk] = 'IVR'
                    if not result.get('has_ivr'):
                        result['has_ivr'] = True

            result = self._validate_result(
                result, detected_speakers, sample_transcript, starts_conversational,
                pre_ivr_map=pre_ivr_map, outbound_agent_spk=outbound_agent_spk,
            )

        print(f"[LLM Corrector] Done in {time.time() - t1:.2f}s | "
              f"IVR={result.get('has_ivr', False)} | "
              f"Roles={result.get('speaker_roles', {})}")

        # ── Step 7: Apply corrected labels to transcript and word alignment ───
        corrected_transcript, corrected_words = self._apply_labels(
            labeled_transcription, aligned_words, result, detected_speakers
        )

        return corrected_transcript, corrected_words, result

    def _get_fallback_result(self, detected_speakers: List[str]) -> Dict:
        """
        Produces a safe default role-assignment result when LLM parsing fails.

        Assigns the first speaker as 'Agent', the second as 'Customer', and any
        additional speakers as 'Agent'.  This mirrors the most common call structure
        (outbound agent calls a single customer).

        Args:
            detected_speakers (List[str]): Ordered list of speaker IDs.

        Returns:
            Dict: A minimal result dict with ``has_ivr=False``, empty ``ivr_text``,
                and the fallback ``speaker_roles`` mapping.
        """
        speaker_roles = {}
        for i, spk in enumerate(detected_speakers):
            speaker_roles[spk] = 'Agent' if i == 0 else 'Customer' if i == 1 else 'Agent'
        return {'has_ivr': False, 'ivr_text': '', 'speaker_roles': speaker_roles, 'reasoning': 'Fallback'}

    def _validate_result(
        self,
        result: Dict,
        detected_speakers: List[str],
        transcript: str,
        starts_conversational: bool,
        pre_ivr_map: Optional[Dict[str, bool]] = None,
        outbound_agent_spk: Optional[str] = None,
    ) -> Dict:
        """
        Applies deterministic business rules to sanitise and override the LLM result.

        Validation steps (in order):
            1. **IVR sanity check**: Rejects IVR claims that are too short, fail the
               pattern match, or contradict a detected conversational start.
            2. **IVR speaker promotion**: Merges ``pre_ivr_map`` and LLM IVR flags;
               rejects IVR for any speaker with more than 2 turns (IVR is always brief).
            3. **Outbound-agent override**: If ``_detect_outbound_agent`` identified an
               agent, forces that speaker to 'Agent' and flips any co-speaker with the
               wrong label.
            4. **Content-based role swap**: Delegates to ``_validate_roles_by_content``
               for phrase-level evidence-based swapping.
            5. **Structural guarantees**: Ensures exactly one Customer exists among
               human speakers (promotes/demotes as needed) and normalises 'Agent_2'
               → 'Agent'.

        Args:
            result (Dict): The LLM result dict (mutated in place).
            detected_speakers (List[str]): All speaker IDs including IVR.
            transcript (str): The (possibly sampled) transcript text.
            starts_conversational (bool): Whether the transcript starts with human speech.
            pre_ivr_map (Optional[Dict[str, bool]]): Pre-classification IVR flags.
            outbound_agent_spk (Optional[str]): Speaker ID forced to Agent by heuristics.

        Returns:
            Dict: The validated (and potentially modified) result dict.
        """
        # Ensure all expected keys exist with safe defaults
        result.setdefault('has_ivr', False)
        result.setdefault('ivr_text', '')
        result.setdefault('speaker_roles', {})

        if pre_ivr_map is None:
            pre_ivr_map = {}

        # ── IVR sanity check ──────────────────────────────────────────────────
        if result.get('has_ivr') and result.get('ivr_text'):
            ivr_text = result['ivr_text']
            if len(ivr_text.strip()) < 10:
                # Too short to be a real IVR announcement
                result['has_ivr'] = False
                result['ivr_text'] = ''
            elif self._is_ivr_like_text(ivr_text):
                print("[LLM Corrector] IVR confirmed by pattern match")
            elif starts_conversational:
                # LLM said IVR but the transcript starts with human speech — reject
                result['has_ivr'] = False
                result['ivr_text'] = ''
        elif result.get('has_ivr') and not result.get('ivr_text'):
            # has_ivr=True but no text supplied — invalid, reset
            result['has_ivr'] = False

        # ── Promote IVR speakers (merge pre-map + LLM labels) ─────────────────
        ivr_speakers = set()
        for spk in detected_speakers:
            if pre_ivr_map.get(spk) or result['speaker_roles'].get(spk) == 'IVR':
                spk_turn_count = transcript.count(f'[{spk}]')
                if spk_turn_count > 2:
                    # IVR never has more than a couple of turns; reject the IVR label
                    print(f"[Validate] Rejecting IVR for '{spk}' — {spk_turn_count} turns")
                    result['speaker_roles'][spk] = 'Customer'
                    continue
                result['speaker_roles'][spk] = 'IVR'
                ivr_speakers.add(spk)

        human_speakers = [spk for spk in detected_speakers if spk not in ivr_speakers]

        # ── Apply outbound-agent override ─────────────────────────────────────
        applied_outbound_override = False
        if outbound_agent_spk and outbound_agent_spk in human_speakers:
            llm_assigned = result['speaker_roles'].get(outbound_agent_spk)
            if llm_assigned != 'Agent':
                print(f"[Outbound] Forcing '{outbound_agent_spk}': {llm_assigned} -> Agent")
                result['speaker_roles'][outbound_agent_spk] = 'Agent'

                other_humans = [s for s in human_speakers if s != outbound_agent_spk]
                if len(other_humans) == 1:
                    # With exactly one other human, they must be the Customer
                    old = result['speaker_roles'].get(other_humans[0])
                    if old != 'Customer':
                        print(f"[Outbound] Forcing '{other_humans[0]}': {old} -> Customer")
                        result['speaker_roles'][other_humans[0]] = 'Customer'
                elif len(other_humans) > 1:
                    # Demote any extra Agents to Customer when the outbound agent is known
                    for s in other_humans:
                        if result['speaker_roles'].get(s) in ('Agent', 'Agent_2'):
                            print(f"[Outbound] Forcing '{s}' Agent -> Customer")
                            result['speaker_roles'][s] = 'Customer'
                applied_outbound_override = True

        # Fill in any missing roles with a safe default
        valid_roles = {'Agent', 'Customer', 'Agent_2'}
        for spk in human_speakers:
            if spk not in result['speaker_roles'] or result['speaker_roles'][spk] not in valid_roles:
                result['speaker_roles'][spk] = 'Agent'

        # ── Content-based role swap (only if outbound override was not applied) ─
        if not applied_outbound_override:
            result = self._validate_roles_by_content(result, human_speakers, transcript)

        # ── Structural guarantees: exactly one Customer ───────────────────────
        agents    = [s for s in human_speakers if result['speaker_roles'][s] in ('Agent', 'Agent_2')]
        customers = [s for s in human_speakers if result['speaker_roles'][s] == 'Customer']

        if len(agents) >= 2 and len(customers) == 0 and len(human_speakers) > 1:
            # All humans labelled Agent — assign the second speaker as Customer
            print(f"[LLM Corrector] {len(agents)} Agents, 0 Customers — assigning second as Customer")
            result['speaker_roles'][human_speakers[1]] = 'Customer'
            customers = [human_speakers[1]]
            agents = [s for s in human_speakers if result['speaker_roles'][s] in ('Agent', 'Agent_2')]

        if len(customers) == 0 and len(human_speakers) > 1:
            # Still no Customer — force the second speaker
            result['speaker_roles'][human_speakers[1]] = 'Customer'
        elif len(customers) > 1:
            # Multiple Customers — merge extras back to Agent
            for spk in customers[1:]:
                result['speaker_roles'][spk] = 'Agent'
                print(f"[LLM Corrector] Multiple Customers — merging {spk} -> Agent")

        # Ensure at least one Agent exists
        agents = [s for s in human_speakers if result['speaker_roles'][s] in ('Agent', 'Agent_2')]
        if len(agents) == 0:
            for spk in human_speakers:
                if result['speaker_roles'][spk] != 'Customer':
                    result['speaker_roles'][spk] = 'Agent'
                    break

        # Normalise 'Agent_2' → 'Agent' (internal intermediate label)
        for spk in detected_speakers:
            if result['speaker_roles'].get(spk) == 'Agent_2':
                result['speaker_roles'][spk] = 'Agent'

        return result

    def _validate_roles_by_content(
        self,
        result: Dict,
        detected_speakers: List[str],
        transcript: str
    ) -> Dict:
        """
        Evidence-based role swap using phrase-level scoring of speaker utterances.

        Scores each of the two primary human speakers on how strongly their
        transcribed speech fits the Agent or Customer pattern.  If the evidence
        suggests the LLM got the roles backwards, the labels are swapped.

        Scoring logic:
            - **Agent score** (applied to the *currently-labelled* Agent's text):
              Phrases like "building office", "I'll forward this to my manager"
              actually indicate the speaker is a Customer (the one who received the
              call), not an Agent — so a high agent-score triggers a swap.
            - **Customer score** (applied to the *currently-labelled* Customer's text):
              Phrases like "I'm calling from", "I'm reaching out" indicate the
              speaker initiated the call, meaning they should be the Agent.

        A swap is triggered when both speakers accumulate evidence pointing in the
        same direction (both score ≥ 2, or one scores ≥ 2 and the other ≥ 1).

        Args:
            result (Dict): The current analysis result dict (mutated in place).
            detected_speakers (List[str]): Human (non-IVR) speaker IDs to evaluate.
            transcript (str): The (possibly sampled) transcript text.

        Returns:
            Dict: The result dict, with speaker_roles potentially swapped.
        """
        if len(detected_speakers) < 2:
            return result

        speaker_roles      = result.get('speaker_roles', {})
        speaker_text       = {spk: [] for spk in detected_speakers}
        speaker_first_line = {}

        # Collect per-speaker utterances and first-line text
        for line in transcript.strip().split('\n'):
            m = re.match(r'\[([^\]]+)\]\s*(.*)', line)
            if m:
                spk  = m.group(1)
                text = m.group(2).strip()
                if spk in speaker_text:
                    speaker_text[spk].append(text.lower())
                    if spk not in speaker_first_line:
                        speaker_first_line[spk] = text.lower()

        # Identify any IVR-like speakers and exclude from human analysis
        ivr_indicators = [
            "thank you for calling", "please wait while", "your call may be recorded",
            "press 1", "this call may be monitored", "for quality and training",
            "please listen carefully", "our menu options have changed",
            "at the tone, please record", "record your message",
            "call has been forwarded", "not available to take your call",
        ]
        ivr_spks = {
            spk for spk in detected_speakers
            if any(p in ' '.join(speaker_text.get(spk, [])) for p in ivr_indicators)
            and len(speaker_text.get(spk, [])) <= 3
        }
        human_speakers = [s for s in detected_speakers if s not in ivr_spks]
        if len(human_speakers) < 2:
            return result

        # Find current Agent and Customer for comparison
        agent_spk = next((s for s in human_speakers if speaker_roles.get(s) == 'Agent'), None)
        cust_spk  = next((s for s in human_speakers if speaker_roles.get(s) == 'Customer'), None)
        if not agent_spk or not cust_spk:
            return result

        # ── Score the current Agent's text for "actually a Customer" evidence ─
        agent_text = ' '.join(speaker_text.get(agent_spk, []))
        a_score, a_matched = 0, set()

        # High-weight phrases (3 pts): department answers = strong Customer signal
        for phrase in ["building office", "billing office", "billing department",
                       "front desk", "reception", "how can i direct your call"]:
            if phrase in agent_text and phrase not in a_matched:
                a_score += 3; a_matched.add(phrase)

        # Medium-weight phrases (2 pts): message-taking and gatekeeping behaviour
        for phrase in [
            "i can take your information", "i can get your information",
            "forward it to my manager", "forward it over to",
            "i don't have his information", "i don't have access",
            "i will give this over to my manager", "i'll forward this",
            "i will get this forwarded", "what is this regarding",
            "what is this in regards to", "what's this about",
            "can i ask what this is about", "is he expecting your call",
            "let me transfer you", "i'll connect you",
            "i can take a message", "can i take a message",
        ]:
            if phrase in agent_text and phrase not in a_matched:
                a_score += 2; a_matched.add(phrase)

        # Low-weight: single-word answers at call start suggest the receiver
        agent_first = speaker_first_line.get(agent_spk, '').strip().rstrip('.,?!').strip()
        if agent_first in ["hello", "yes", "speaking", "hi", "yeah"]:
            a_score += 1

        # ── Score the current Customer's text for "actually an Agent" evidence ─
        cust_text = ' '.join(speaker_text.get(cust_spk, []))
        c_score, c_matched = 0, set()

        # High-weight phrases (3 pts): explicit company/affiliation mentions
        for phrase in [
            "i'm with ", "i am with ", "i'm from ", "i am from ",
            "calling from ", "calling you from ",
            "i'm reaching out", "i am reaching out",
            "i'm looking to speak with", "i'm trying to reach",
        ]:
            if phrase in cust_text and phrase not in c_matched:
                c_score += 3; c_matched.add(phrase)

        # Medium-weight phrases (2 pts): outbound-call framing language
        for phrase in [
            "the reason for my call", "the purpose of my call",
            "i wanted to reach out", "i wanted to follow up",
            "do you have a few minutes", "is this a good time",
            "i'd like to schedule", "i'd like to set up", "on behalf of",
        ]:
            if phrase in cust_text and phrase not in c_matched:
                c_score += 2; c_matched.add(phrase)

        # Low-weight: outbound framing in the very first utterance
        cust_first = speaker_first_line.get(cust_spk, '').strip()
        if any(p in cust_first for p in ["i'm with", "i'm from", "calling from"]):
            c_score += 1

        # Swap roles if both speakers have accumulated sufficient contrary evidence
        should_swap = (
            (a_score >= 2 and c_score >= 2)
            or (a_score >= 2 and c_score >= 1)
            or (c_score >= 2 and a_score >= 1)
        )
        if should_swap:
            print(f"[ContentValidator] SWAPPING: {agent_spk}->Customer, {cust_spk}->Agent")
            result['speaker_roles'][agent_spk] = 'Customer'
            result['speaker_roles'][cust_spk]  = 'Agent'

        return result

    def _post_check_ivr_speakers(
        self, aligned_words: List[Dict], labeled_transcription: str
    ) -> List[Dict]:
        """
        Post-processing pass to catch any remaining IVR speakers in word-level data.

        After the main label-application step, this method scans the ``aligned_words``
        list for speakers that:
            1. Have fewer than 100 words total (IVR is always brief).
            2. Only appear within the first 15% of the word list (IVR is always first).
            3. Produce IVR-like text according to ``_is_ivr_like_text``.

        If relabelling these speakers would leave fewer than two human speakers in the
        word list, the relabelling is skipped to avoid producing an invalid transcript.

        Args:
            aligned_words (List[Dict]): Word-level alignment data with ``"speaker"``
                and ``"text"`` keys.
            labeled_transcription (str): The original (uncorrected) transcript text,
                used for context (not directly parsed here).

        Returns:
            List[Dict]: The (possibly modified) aligned words list, with any detected
                IVR speakers relabelled to ``'IVR'``.  The original speaker ID is
                preserved in an ``'original_speaker'`` field.
        """
        if not aligned_words:
            return aligned_words

        # Build a word-list per speaker for IVR phrase matching
        speaker_words: Dict[str, List[str]] = {}
        for w in aligned_words:
            spk = w.get('speaker', '')
            speaker_words.setdefault(spk, []).append(w.get('text', ''))

        speakers_to_relabel = []
        for spk, words in speaker_words.items():
            if spk == 'IVR' or len(words) > 100:
                continue

            # Find the positional range of this speaker in the word list
            positions = [i for i, w in enumerate(aligned_words) if w.get('speaker') == spk]
            if not positions:
                continue

            # Only consider speakers that appear exclusively in the first 15% of the call
            if max(positions) > len(aligned_words) * 0.15:
                continue

            if self._is_ivr_like_text(' '.join(words).lower()):
                speakers_to_relabel.append(spk)

        if not speakers_to_relabel:
            return aligned_words

        # Safety check: ensure at least 2 human speakers would remain after relabelling
        remaining_human = {
            w.get('speaker', '') for w in aligned_words
            if w.get('speaker') != 'IVR' and w.get('speaker') not in speakers_to_relabel
        }
        if len(remaining_human) < 2:
            print(f"[Post-IVR] Skipping — only {len(remaining_human)} human(s) would remain")
            return aligned_words

        for spk in speakers_to_relabel:
            print(f"[Post-IVR] Relabeling '{spk}' -> IVR")
            for w in aligned_words:
                if w.get('speaker') == spk:
                    w['original_speaker'] = spk  # Preserve for auditability
                    w['speaker'] = 'IVR'
        return aligned_words

    def _apply_labels(
        self,
        labeled_transcription: str,
        aligned_words: Optional[List[Dict]],
        analysis_result: Dict,
        detected_speakers: List[str]
    ) -> Tuple[str, Optional[List[Dict]]]:
        """
        Rewrites the transcript and aligned-word data with the validated speaker labels.

        This is the final transformation step.  It:
            1. Builds a ``label_map`` from the validated ``analysis_result``.
            2. If ``aligned_words`` are provided, relabels each word's ``"speaker"``
               field and marks IVR words using ``_mark_ivr_words``.
            3. Runs a final ``_post_check_ivr_speakers`` pass on the word data.
            4. Rebuilds the transcript string from the corrected word data (preferred)
               or by string-replacing the old speaker tags in the original transcript.

        Args:
            labeled_transcription (str): The original transcript with raw diarization
                labels (e.g., ``[speaker_0]``).
            aligned_words (Optional[List[Dict]]): Word-level alignment data.
            analysis_result (Dict): Validated result dict from ``_validate_result``.
            detected_speakers (List[str]): All original speaker IDs.

        Returns:
            Tuple[str, Optional[List[Dict]]]:
                - The corrected transcript string.
                - The corrected aligned-word list (or ``None`` if not provided).
        """
        speaker_roles = analysis_result.get('speaker_roles', {})
        ivr_text = analysis_result.get('ivr_text', '').strip()
        has_ivr  = analysis_result.get('has_ivr', False) and len(ivr_text) > 10

        # Build the old_label → new_label map; normalise Agent_2 → Agent
        label_map = {spk: speaker_roles.get(spk, 'Unknown') for spk in detected_speakers}
        print(f"[LLM Corrector] Label map: {label_map}")

        for spk in label_map:
            if label_map[spk] == 'Agent_2':
                label_map[spk] = 'Agent'

        # ── Relabel aligned words ─────────────────────────────────────────────
        corrected_words = None
        if aligned_words:
            corrected_words = [
                word.copy() for word in aligned_words
                if isinstance(word, dict) and 'text' in word
            ]

            # Mark IVR words before applying human-speaker role labels
            if has_ivr:
                corrected_words = self._mark_ivr_words(corrected_words, ivr_text)

            for word in corrected_words:
                if word.get('speaker') != 'IVR':
                    old = word.get('speaker', '')
                    if old in label_map:
                        word['speaker'] = label_map[old]
                    else:
                        # Fuzzy match: handle slight speaker-ID format differences
                        matched = False
                        for ds, role in label_map.items():
                            if ds in old or old in ds:
                                word['speaker'] = role
                                matched = True
                                break
                        if not matched and old not in ['Agent', 'Customer', 'Agent_2', 'IVR']:
                            print(f"[LLM Corrector] Warning: Unknown speaker '{old}'")

            # Second pass: catch any words whose speaker was mapped to IVR
            for word in corrected_words:
                spk = word.get('speaker', '')
                if label_map.get(spk) == 'IVR' or speaker_roles.get(spk) == 'IVR':
                    word['original_speaker'] = spk
                    word['speaker'] = 'IVR'

            corrected_words = self._post_check_ivr_speakers(corrected_words, labeled_transcription)

        # ── Rebuild the transcript string ─────────────────────────────────────
        if corrected_words:
            # Preferred path: rebuild from corrected word-level data
            corrected_transcription = self._rebuild_transcription(corrected_words)
        else:
            # Fallback: string-replace speaker tags directly in the original transcript
            corrected_transcription = labeled_transcription
            if has_ivr:
                corrected_transcription = self._mark_ivr_in_text(
                    corrected_transcription, ivr_text, label_map
                )
            else:
                for old_label, new_label in label_map.items():
                    corrected_transcription = corrected_transcription.replace(
                        f'[{old_label}]', f'[{new_label}]'
                    )

        return corrected_transcription, corrected_words

    def _mark_ivr_in_text(self, transcription: str, ivr_text: str, label_map: Dict) -> str:
        """
        Replaces the speaker tag on the IVR line(s) in a plain-text transcript.

        Iterates through transcript lines and relabels the first line whose text
        matches the IVR text (using ``_text_matches_ivr``) as ``[IVR]``.  All
        subsequent lines are relabelled using the ``label_map``.

        Used as a fallback when no word-level alignment data is available.

        Args:
            transcription (str): Original transcript with raw diarization labels.
            ivr_text (str): The IVR text identified by the LLM.
            label_map (Dict[str, str]): Mapping of raw speaker ID to corrected role.

        Returns:
            str: Transcript with the IVR line relabelled as ``[IVR]``.
        """
        lines = transcription.split('\n')
        result_lines = []
        ivr_text_lower = ivr_text.lower().strip()
        ivr_marked = False  # Only relabel the first matching IVR line

        for line in lines:
            m = re.match(r'\[([^\]]+)\]\s*(.*)', line)
            if not m:
                result_lines.append(line)
                continue
            speaker = m.group(1)
            text    = m.group(2).strip()

            if not ivr_marked and self._text_matches_ivr(text.lower(), ivr_text_lower):
                result_lines.append(f'[IVR] {text}')
                ivr_marked = True
                continue
            result_lines.append(f'[{label_map.get(speaker, speaker)}] {text}')
        return '\n'.join(result_lines)

    def _text_matches_ivr(self, text: str, ivr_text: str) -> bool:
        """
        Checks whether a transcript line's text matches the identified IVR text.

        Uses two matching strategies:
            1. **Substring match**: Either string fully contains the other.
            2. **Jaccard-like word overlap**: At least 60% of the words in the
               shorter string appear in the longer string (handles minor ASR errors
               and word reordering).

        Args:
            text (str): Lowercased text of a single transcript line.
            ivr_text (str): Lowercased IVR text identified by the LLM.

        Returns:
            bool: ``True`` if the texts are considered a match.
        """
        text = text.strip(); ivr_text = ivr_text.strip()
        if not text or not ivr_text:
            return False

        # Exact substring containment
        if text in ivr_text or ivr_text in text:
            return True

        # Word-overlap similarity (≥ 60% of shorter text's words appear in the other)
        tw = set(text.split()); iw = set(ivr_text.split())
        if not tw or not iw:
            return False
        return len(tw & iw) / min(len(tw), len(iw)) >= 0.6

    def _mark_ivr_words(self, aligned_words: List[Dict], ivr_text: str) -> List[Dict]:
        """
        Marks word-level alignment entries as IVR based on IVR text matching.

        Accumulates word text character by character until either:
            - The accumulated text contains the full IVR string, or
            - The word count equals the IVR word count and ≥ 80% of IVR words are
              present (handles minor ASR transcription differences).

        All words up to and including the matching index are relabelled as ``'IVR'``,
        with the original speaker preserved in ``'original_speaker'``.

        Args:
            aligned_words (List[Dict]): Word-level alignment data to modify in place.
            ivr_text (str): The IVR text string to match against.

        Returns:
            List[Dict]: The modified aligned words list.
        """
        if not ivr_text or not aligned_words:
            return aligned_words

        ivr_lower  = ivr_text.lower()
        ivr_wlist  = ivr_lower.split()
        accumulated = ""
        ivr_end_idx = -1

        for i, word in enumerate(aligned_words):
            accumulated += word.get('text', '').lower() + " "

            # Strategy 1: IVR text is fully contained in accumulated transcript
            if ivr_lower in accumulated.strip():
                ivr_end_idx = i; break

            # Strategy 2: Word count reached — check overlap ratio
            if len(accumulated.split()) >= len(ivr_wlist):
                acc_w = set(accumulated.lower().split())
                if len(acc_w & set(ivr_wlist)) / len(ivr_wlist) >= 0.8:
                    ivr_end_idx = i; break

        if ivr_end_idx >= 0:
            for i in range(ivr_end_idx + 1):
                aligned_words[i]['original_speaker'] = aligned_words[i].get('speaker', '')
                aligned_words[i]['speaker'] = 'IVR'
        return aligned_words

    def _rebuild_transcription(self, aligned_words: List[Dict]) -> str:
        """
        Reconstructs a labelled transcript string from word-level alignment data.

        Groups consecutive words that share the same ``"speaker"`` value into a
        single ``[Speaker] word1 word2 ...`` line.  Speaker changes produce a
        new line with the new speaker's label.

        Args:
            aligned_words (List[Dict]): Word-level alignment data with ``"speaker"``
                and ``"text"`` keys.

        Returns:
            str: Multi-line transcript string in ``[Speaker] text`` format.
        """
        if not aligned_words:
            return ""

        result = []
        current_speaker = None
        current_words   = []

        for word_info in aligned_words:
            speaker = word_info.get('speaker', 'Unknown')
            text    = word_info.get('text', '')

            if speaker != current_speaker:
                # Flush the current speaker's buffered words before switching
                if current_words and current_speaker:
                    result.append(f"[{current_speaker}] {' '.join(current_words)}")
                current_speaker = speaker
                current_words   = [text]
            else:
                current_words.append(text)

        # Flush the final speaker's buffered words
        if current_words and current_speaker:
            result.append(f"[{current_speaker}] {' '.join(current_words)}")

        return '\n'.join(result)

    async def assign_agent_customer_labels(
        self,
        labeled_transcription: str,
        request_id: str,
        aligned_words: Optional[List[Dict]] = None,
        sample_size: int = 30
    ) -> Tuple[str, Optional[List[Dict]]]:
        """
        Convenience wrapper around ``analyze_transcript`` that discards the metadata.

        Callers that only need the corrected transcript and word list (and do not
        need the raw LLM result dict) can use this method to keep their code
        simpler.

        Args:
            labeled_transcription (str): Multi-line transcript with ``[speaker_N]``
                prefixes.
            request_id (str): Unique request identifier for Triton tracing.
            aligned_words (Optional[List[Dict]]): Word-level alignment data.
            sample_size (int): Maximum lines sent to the LLM.  Defaults to 30.

        Returns:
            Tuple[str, Optional[List[Dict]]]:
                - The corrected transcript string.
                - The corrected aligned-word list (or ``None``).
        """
        corrected_transcript, corrected_words, _ = await self.analyze_transcript(
            labeled_transcription, request_id, aligned_words, sample_size
        )
        return corrected_transcript, corrected_words

    async def close(self):
        """
        Gracefully closes the Triton gRPC client connection.

        Should be called when the corrector is no longer needed to release the
        underlying gRPC channel and avoid resource leaks.  After calling ``close``,
        call ``initialize`` before making any further inference requests.
        """
        if self.client:
            await self.client.close()
            self.client = None
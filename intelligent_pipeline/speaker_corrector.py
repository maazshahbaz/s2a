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
    Uses Mistral-Nemo LLM to:
    1. Detect IVR/automated system messages in transcripts
    2. Assign Agent/Customer labels to human speakers
    """
    
    def __init__(self, triton_url: str = None, model_name: str = None):
        self.triton_url = triton_url or 'localhost:3701'
        self.model_name = model_name or 'mistral-nemo'
        self.client = None
        self.client = grpcclient_aio.InferenceServerClient(url=self.triton_url)
        print(f"[LLM Corrector] Initialized with Triton at {self.triton_url}")
    
    def _get_system_prompt(self) -> str:
        """System prompt for IVR detection and speaker labeling."""
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

CRITICAL IDENTIFICATION RULES (in PRIORITY ORDER — earlier rules override later ones):

### HIGHEST PRIORITY — these ALWAYS identify the AGENT:
1. Says "I'm with [Company]" or "I'm from [Company]" or "This is [Name] from [Company]"
2. Says "I'm calling from [Company]" or "I'm calling you from [Company]"
3. Says "I'm looking to speak with [Name]" or "I'm trying to reach [Name]"
4. Says "I'm reaching out to you about..." or "I'm calling regarding..."
5. Explains their PURPOSE for calling ("regarding the phone system", "about retired technology")

### HIGHEST PRIORITY — these ALWAYS identify the CUSTOMER:
1. Answers the phone with a department name: "Building Office", "Billing Office", "Front Desk"
2. Says "I can take your information and forward it to my manager"
3. Says "What is this regarding?" or "Is he helping you with something?" (screening/gatekeeping)
4. Takes a message: "I'll forward this to...", "I'll give this to my manager"

### SECONDARY RULES (only if the above rules don't apply):
5. The Agent may ask diagnostic questions: "When did you purchase...?", "Do you have an iPhone?"
6. The Customer may describe their problem: "the monitor isn't working", "I had issues with..."
7. The Customer may provide personal info when asked by Agent during tech support calls

### IMPORTANT — DO NOT CONFUSE:
- A RECEPTIONIST asking screening questions ("What's your name?", "Which department?") is the CUSTOMER, not the Agent. Screening questions are NOT diagnostic questions.
- The person who provides their company name and reason for calling is the AGENT, even if they also provide their personal name/number when asked by the receptionist.

### CALL TYPE PATTERNS:

INBOUND SUPPORT CALL (Agent answers):
- Agent: "Thanks for calling. How can I help?" → AGENT
- Customer: "I bought a computer and it's not working" → CUSTOMER

OUTBOUND / COLD CALL (Agent calls in):
- Receptionist: "Building Office" or "Hello?" → CUSTOMER  
- Caller: "Hi, I'm looking to speak with Mike. I'm with Talk Loop." → AGENT
- Receptionist: "What's your name? And a phone number?" → CUSTOMER (taking a message, NOT asking diagnostic questions)
- Caller: "My name is Cody. 612-355-8393." → AGENT (providing info to receptionist)

IMPORTANT - HANDLING 3+ SPEAKERS:
Most customer service calls have exactly 2 people: 1 Agent and 1 Customer.
If you see 3 or more speakers, this is usually a DIARIZATION ERROR where one person's speech was incorrectly split.

When you see 3 speakers:
- Look at the CONTENT of what each speaker says
- If two speakers are both asking questions or both providing service → they are likely the SAME AGENT
- If two speakers are both describing problems or answering questions → they are likely the SAME CUSTOMER
- Assign the same role to speakers who appear to be the same person

## RESPONSE FORMAT

Return ONLY a JSON object:
{
    "has_ivr": true/false,
    "ivr_text": "exact IVR text from transcript or empty string",
    "speaker_roles": {"speaker_0": "Agent" or "Customer", "speaker_1": "Agent" or "Customer", ...},
    "reasoning": "brief explanation of why you assigned each role"
}"""
    
    async def initialize(self):
        self.client = grpcclient_aio.InferenceServerClient(url=self.triton_url)
    
    def _format_prompt(self, user_prompt: str) -> str:
        system_prompt = self._get_system_prompt()
        return f"[INST] {system_prompt}\n\n{user_prompt} [/INST]"
    
    async def _generate_async(self, prompt: str, request_id: str = None) -> str:
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        try:
            input_data = np.array([[prompt]], dtype=object)
            inputs = [grpcclient_aio.InferInput("prompt", [1, 1], "BYTES")]
            inputs[0].set_data_from_numpy(input_data)
            outputs = [grpcclient_aio.InferRequestedOutput("generated_text")]
            
            response = await self.client.infer(
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs,
                request_id=request_id
            )
            
            output_text = response.as_numpy("generated_text")[0].decode('utf-8')
            return output_text
        except Exception as e:
            print(f"[LLM Corrector] Error during Triton inference: {e}")
            raise
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict]:
        """Extract JSON from LLM response."""
        # Try ```json blocks
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object
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
        """Extract unique speaker labels from transcript."""
        speaker_pattern = re.compile(r'\[(speaker_\d+)\]')
        matches = speaker_pattern.findall(labeled_transcription)
        seen = set()
        speakers = []
        for match in matches:
            if match not in seen:
                seen.add(match)
                speakers.append(match)
        return speakers
    
    def _is_ivr_like_text(self, text: str) -> bool:
        """Check if text has IVR characteristics."""
        text_lower = text.lower()
        
        # IVR indicators
        ivr_phrases = [
            "thank you for calling",
            "please wait while",
            "your call may be recorded",
            "press 1",
            "press 2",
            "please hold",
            "home of the",  # Common in company slogans
            "fixture-free guarantee",
        ]
        
        return any(phrase in text_lower for phrase in ivr_phrases)
    
    def _is_conversational_start(self, transcript: str) -> bool:
        """Check if transcript starts with human conversation (not IVR)."""
        lines = transcript.strip().split('\n')
        if not lines:
            return True
        
        first_line = lines[0].lower()
        
        # Pure conversational starters (definitely NOT IVR)
        conversational_starters = [
            "hello.",
            "hi,",
            "hi.",
            "good afternoon",
            "good morning",
            "good evening",
        ]
        
        # Check if first line is a simple greeting
        for starter in conversational_starters:
            if starter in first_line and not self._is_ivr_like_text(first_line):
                return True
        
        return False
    
    async def analyze_transcript(
        self,
        labeled_transcription: str,
        request_id: str,
        aligned_words: Optional[List[Dict]] = None,
        sample_size: int = 30
    ) -> Tuple[str, Optional[List[Dict]], Dict]:
        """Analyze transcript for IVR detection and speaker role assignment."""
        t1 = time.time()
        
        # Detect speakers
        detected_speakers = self._detect_speakers_in_transcript(labeled_transcription)
        
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
        
        # Take sample for analysis
        lines = labeled_transcription.split('\n')
        sample_transcript = '\n'.join(lines[:sample_size]) if len(lines) > sample_size else labeled_transcription
        
        # Quick check: if transcript starts conversationally, likely no IVR
        starts_conversational = self._is_conversational_start(sample_transcript)
        
        # Build prompt
        speaker_roles_format = ', '.join([f'"{spk}": "Agent" or "Customer"' for spk in detected_speakers])
        
        user_prompt = f"""Analyze this transcript:

TRANSCRIPT:
{sample_transcript}

IMPORTANT - IDENTIFICATION PRIORITY:
1. HIGHEST PRIORITY: If someone says "I'm with [Company]", "I'm from [Company]", "calling from [Company]", or "I'm looking to speak with [Name]" → they are ALWAYS the AGENT
2. HIGHEST PRIORITY: If someone answers with a department name ("Building Office", "Billing Office") or says "I'll forward this to my manager" → they are ALWAYS the CUSTOMER
3. A receptionist asking "What's your name?" or "What is this about?" is a CUSTOMER doing screening, NOT an Agent asking diagnostic questions
4. In inbound support calls: the person who says "Thanks for calling, how can I help?" is the AGENT
5. The person describing their problem ("the monitor isn't working") is the CUSTOMER

Return JSON with:
- has_ivr: true if there's automated IVR at the START (false if it starts with human conversation)
- ivr_text: the exact IVR text (empty string if no IVR)
- speaker_roles: {{{speaker_roles_format}}}
- reasoning: brief explanation of why you assigned each role"""
        
        full_prompt = self._format_prompt(user_prompt)
        
        print("[LLM Corrector] Analyzing transcript...")
        response = await self._generate_async(full_prompt, request_id)
        
        # Parse response
        result = self._extract_json_from_response(response)
        
        if result is None:
            print("[LLM Corrector] Warning: Could not parse LLM response")
            result = self._get_fallback_result(detected_speakers)
        else:
            result = self._validate_result(result, detected_speakers, sample_transcript, starts_conversational)
        
        # Log results
        print(f"[LLM Corrector] Analysis complete in {time.time() - t1:.2f}s:")
        print(f"  IVR detected: {result.get('has_ivr', False)}")
        if result.get('ivr_text'):
            print(f"  IVR text: \"{result['ivr_text'][:80]}...\"" if len(result.get('ivr_text', '')) > 80 else f"  IVR text: \"{result.get('ivr_text')}\"")
        print(f"  Speaker roles: {result.get('speaker_roles', {})}")
        
        # Apply labels
        corrected_transcript, corrected_words = self._apply_labels(
            labeled_transcription, aligned_words, result, detected_speakers
        )
        
        return corrected_transcript, corrected_words, result
    
    def _get_fallback_result(self, detected_speakers: List[str]) -> Dict:
        """Fallback when LLM parsing fails."""
        speaker_roles = {}
        for i, spk in enumerate(detected_speakers):
            speaker_roles[spk] = 'Agent' if i == 0 else 'Customer' if i == 1 else 'Agent'
        return {
            'has_ivr': False,
            'ivr_text': '',
            'speaker_roles': speaker_roles,
            'reasoning': 'Fallback: default assignment'
        }
    
    def _validate_result(self, result: Dict, detected_speakers: List[str], transcript: str, starts_conversational: bool) -> Dict:
        """Validate and fix the analysis result."""
        
        # Ensure required fields
        result.setdefault('has_ivr', False)
        result.setdefault('ivr_text', '')
        result.setdefault('speaker_roles', {})
        
        # If transcript starts conversationally and IVR text doesn't look like IVR, disable it
        if starts_conversational and result.get('has_ivr'):
            ivr_text = result.get('ivr_text', '')
            if not self._is_ivr_like_text(ivr_text):
                print("[LLM Corrector] Transcript starts conversationally, disabling IVR")
                result['has_ivr'] = False
                result['ivr_text'] = ''
        
        # Validate IVR text actually looks like IVR
        if result.get('has_ivr') and result.get('ivr_text'):
            if not self._is_ivr_like_text(result['ivr_text']):
                print("[LLM Corrector] IVR text doesn't match IVR patterns, disabling")
                result['has_ivr'] = False
                result['ivr_text'] = ''
        
        # Ensure all speakers have valid roles
        valid_roles = {'Agent', 'Customer', 'Agent_2'}
        for spk in detected_speakers:
            if spk not in result['speaker_roles'] or result['speaker_roles'][spk] not in valid_roles:
                result['speaker_roles'][spk] = 'Agent'
        
        # FIX: Content-based role validation - detect and correct role swaps
        result = self._validate_roles_by_content(result, detected_speakers, transcript)
        
        # Count roles
        agents = [spk for spk in detected_speakers if result['speaker_roles'][spk] in ['Agent', 'Agent_2']]
        customers = [spk for spk in detected_speakers if result['speaker_roles'][spk] == 'Customer']
        
        # Handle case: multiple agents but no/few customers (likely diarization error)
        # In a typical 2-person call, we need exactly 1 Agent and 1 Customer
        if len(agents) >= 2 and len(customers) == 0:
            # No customer assigned - assign the speaker with most words in transcript as one role
            # and the rest as the other
            print(f"[LLM Corrector] Warning: {len(agents)} Agents and 0 Customers - assigning second speaker as Customer")
            if len(detected_speakers) > 1:
                result['speaker_roles'][detected_speakers[1]] = 'Customer'
                customers = [detected_speakers[1]]
                agents = [spk for spk in detected_speakers if result['speaker_roles'][spk] in ['Agent', 'Agent_2']]
        
        # Ensure exactly one Customer
        if len(customers) == 0 and len(detected_speakers) > 1:
            result['speaker_roles'][detected_speakers[1]] = 'Customer'
        elif len(customers) > 1:
            # Keep only the first customer, merge others to Agent
            for spk in customers[1:]:
                result['speaker_roles'][spk] = 'Agent'
                print(f"[LLM Corrector] Multiple Customers detected, merging {spk} to Agent")
        
        # Ensure at least one Agent
        agents = [spk for spk in detected_speakers if result['speaker_roles'][spk] in ['Agent', 'Agent_2']]
        if len(agents) == 0:
            for spk in detected_speakers:
                if result['speaker_roles'][spk] != 'Customer':
                    result['speaker_roles'][spk] = 'Agent'
                    break
        
        # Final cleanup: merge Agent_2 to Agent if no explicit transfer
        for spk in detected_speakers:
            if result['speaker_roles'][spk] == 'Agent_2':
                result['speaker_roles'][spk] = 'Agent'
        
        return result
    
    def _validate_roles_by_content(
        self, 
        result: Dict, 
        detected_speakers: List[str], 
        transcript: str
    ) -> Dict:
        """
        Content-based validation: detect and correct role swaps using contradiction detection.
        
        Only swaps roles when BOTH conditions are met:
        1. The speaker labeled Agent has strong CUSTOMER signals (receptionist/gatekeeping)
        2. The speaker labeled Customer has strong AGENT signals (company identification)
        
        This conservative approach prevents false swaps on ambiguous calls.
        Works with any number of speakers (2, 3, etc.) by finding the Agent and Customer.
        """
        if len(detected_speakers) < 2:
            return result
        
        speaker_roles = result.get('speaker_roles', {})
        
        # Build per-speaker text first (needed to identify IVR speaker)
        import re
        speaker_text = {spk: [] for spk in detected_speakers}
        lines = transcript.strip().split('\n')
        
        for line in lines:
            match = re.match(r'\[([^\]]+)\]\s*(.*)', line)
            if match:
                spk = match.group(1)
                text = match.group(2).strip()
                if spk in speaker_text:
                    speaker_text[spk].append(text.lower())
        
        # Identify and exclude the IVR speaker from role checking
        # IVR speaker's text contains automated phrases like "thank you for calling", "please wait"
        ivr_speakers = set()
        ivr_indicators = ["thank you for calling", "please wait while", "your call may be recorded",
                          "press 1", "please hold", "home of the"]
        
        for spk in detected_speakers:
            all_text = ' '.join(speaker_text.get(spk, []))
            # A speaker is IVR if they have IVR phrases AND very few lines (typically 1-2)
            has_ivr_phrases = any(phrase in all_text for phrase in ivr_indicators)
            few_lines = len(speaker_text.get(spk, [])) <= 2
            if has_ivr_phrases and few_lines:
                ivr_speakers.add(spk)
                print(f"[Content Validator] Excluding IVR speaker: {spk}")
        
        # Find Agent and Customer among non-IVR speakers
        human_speakers = [spk for spk in detected_speakers if spk not in ivr_speakers]
        
        if len(human_speakers) < 2:
            print(f"[Content Validator] Only {len(human_speakers)} human speaker(s) after IVR exclusion, skipping")
            return result
        
        agent_spk = None
        customer_spk = None
        for spk in human_speakers:
            role = speaker_roles.get(spk, '')
            if role == 'Agent' and agent_spk is None:
                agent_spk = spk
            elif role == 'Customer' and customer_spk is None:
                customer_spk = spk
        
        if not agent_spk or not customer_spk:
            print(f"[Content Validator] Skipping - need both Agent and Customer "
                  f"(found agent={agent_spk}, customer={customer_spk})")
            return result
        
        print(f"[Content Validator] Checking roles: {agent_spk}=Agent, {customer_spk}=Customer "
              f"(total speakers: {len(detected_speakers)})")
        
        # Check if assigned Agent has strong CUSTOMER signals
        agent_text = ' '.join(speaker_text.get(agent_spk, []))
        agent_has_customer_signals = False
        
        customer_receptionist_phrases = [
            "building office", "billing office", "billing department",
            "front desk", "reception",
        ]
        customer_gatekeeping_phrases = [
            "i can take your information", "i can get your information",
            "forward it to my manager", "forward it over to",
            "i don't have his information", "i don't have access",
            "i will give this over to my manager",
            "i'll forward this", "i will get this forwarded",
        ]
        
        for phrase in customer_receptionist_phrases + customer_gatekeeping_phrases:
            if phrase in agent_text:
                agent_has_customer_signals = True
                break
        
        # Check if assigned Customer has strong AGENT signals (company identification)
        customer_text = ' '.join(speaker_text.get(customer_spk, []))
        customer_has_agent_signals = False
        
        agent_company_phrases = [
            "i'm with ", "i am with ", "i'm from ", "i am from ",
            "calling from ", "calling you from ",
            "this is {} from ", "this is {} with ",
        ]
        
        for phrase in agent_company_phrases:
            if '{}' in phrase:
                base = phrase.split('{}')[0]
                if base in customer_text:
                    customer_has_agent_signals = True
                    break
            elif phrase in customer_text:
                customer_has_agent_signals = True
                break
        
        # Only swap if BOTH contradictions are present
        if agent_has_customer_signals and customer_has_agent_signals:
            print(f"[Content Validator] ROLE SWAP DETECTED - contradictions on BOTH sides:")
            print(f"  '{agent_spk}' labeled Agent but has Customer signals (receptionist/gatekeeping)")
            print(f"  '{customer_spk}' labeled Customer but has Agent signals (company identification)")
            print(f"  Swapping: {agent_spk}→Customer, {customer_spk}→Agent")
            
            result['speaker_roles'][agent_spk] = 'Customer'
            result['speaker_roles'][customer_spk] = 'Agent'
        elif agent_has_customer_signals or customer_has_agent_signals:
            # Only one side has contradiction — log but don't swap (ambiguous)
            print(f"[Content Validator] Partial contradiction detected (not swapping):")
            if agent_has_customer_signals:
                print(f"  '{agent_spk}' (Agent) has some Customer-like signals")
            if customer_has_agent_signals:
                print(f"  '{customer_spk}' (Customer) has some Agent-like signals")
        else:
            print(f"[Content Validator] No contradictions - roles confirmed")
        
        return result
    
    def _apply_labels(
        self,
        labeled_transcription: str,
        aligned_words: Optional[List[Dict]],
        analysis_result: Dict,
        detected_speakers: List[str]
    ) -> Tuple[str, Optional[List[Dict]]]:
        """Apply labels to transcript and aligned words."""
        
        speaker_roles = analysis_result.get('speaker_roles', {})
        ivr_text = analysis_result.get('ivr_text', '').strip()
        has_ivr = analysis_result.get('has_ivr', False) and len(ivr_text) > 10
        
        # Build label mapping from detected speakers
        label_map = {spk: speaker_roles.get(spk, 'Unknown') for spk in detected_speakers}
        print(f"[LLM Corrector] Label map: {label_map}")
        
        # Check for multiple speakers with the same role (diarization error)
        # Merge Agent_2 into Agent, and any duplicate roles
        role_counts = {}
        for spk, role in label_map.items():
            role_counts[role] = role_counts.get(role, 0) + 1
        
        # Normalize Agent_2 to Agent for merging purposes
        for spk in label_map:
            if label_map[spk] == 'Agent_2':
                label_map[spk] = 'Agent'
                print(f"[LLM Corrector] Merging {spk} (Agent_2) into Agent")
        
        print(f"[LLM Corrector] Final label map after merge: {label_map}")
        
        # Process aligned words
        corrected_words = None
        if aligned_words:
            corrected_words = [word.copy() for word in aligned_words if isinstance(word, dict) and 'text' in word]
            
            # Debug: Show speaker distribution in input aligned_words
            input_speaker_counts = {}
            for w in corrected_words:
                spk = w.get('speaker', 'Unknown')
                input_speaker_counts[spk] = input_speaker_counts.get(spk, 0) + 1
            print(f"[LLM Corrector] Input aligned_words speaker distribution: {input_speaker_counts}")
            
            # Mark IVR words at beginning
            if has_ivr:
                corrected_words = self._mark_ivr_words(corrected_words, ivr_text)
            
            # Apply speaker roles to non-IVR words
            for word in corrected_words:
                if word.get('speaker') != 'IVR':
                    old_speaker = word.get('speaker', '')
                    if old_speaker in label_map:
                        word['speaker'] = label_map[old_speaker]
                    else:
                        # Speaker not in label_map - might be using different format
                        matched = False
                        for detected_spk, role in label_map.items():
                            if detected_spk in old_speaker or old_speaker in detected_spk:
                                word['speaker'] = role
                                matched = True
                                break
                        
                        if not matched and old_speaker not in ['Agent', 'Customer', 'Agent_2', 'IVR']:
                            print(f"[LLM Corrector] Warning: Unknown speaker '{old_speaker}' not in label_map")
            
            # Debug: Show speaker distribution after mapping
            output_speaker_counts = {}
            for w in corrected_words:
                spk = w.get('speaker', 'Unknown')
                output_speaker_counts[spk] = output_speaker_counts.get(spk, 0) + 1
            print(f"[LLM Corrector] Output speaker distribution: {output_speaker_counts}")
        
        # Rebuild transcription from corrected words
        if corrected_words:
            corrected_transcription = self._rebuild_transcription(corrected_words)
        else:
            # No aligned words - work directly with text
            corrected_transcription = labeled_transcription
            
            # Handle IVR marking in text
            if has_ivr:
                corrected_transcription = self._mark_ivr_in_text(
                    corrected_transcription, ivr_text, label_map
                )
            else:
                # Just replace speaker labels
                for old_label, new_label in label_map.items():
                    corrected_transcription = corrected_transcription.replace(f'[{old_label}]', f'[{new_label}]')
        
        return corrected_transcription, corrected_words
    
    def _mark_ivr_in_text(
        self,
        transcription: str,
        ivr_text: str,
        label_map: Dict[str, str]
    ) -> str:
        """Mark IVR in text-based transcription and apply other labels."""
        lines = transcription.split('\n')
        result_lines = []
        ivr_text_lower = ivr_text.lower().strip()
        ivr_marked = False
        
        for line in lines:
            # Extract speaker and text from line
            match = re.match(r'\[([^\]]+)\]\s*(.*)', line)
            if not match:
                result_lines.append(line)
                continue
            
            speaker = match.group(1)
            text = match.group(2).strip()
            text_lower = text.lower()
            
            # Check if this line is part of the IVR
            if not ivr_marked:
                # Check if this text is similar to the IVR text
                if self._text_matches_ivr(text_lower, ivr_text_lower):
                    result_lines.append(f'[IVR] {text}')
                    ivr_marked = True
                    continue
            
            # Apply regular label mapping
            new_speaker = label_map.get(speaker, speaker)
            result_lines.append(f'[{new_speaker}] {text}')
        
        return '\n'.join(result_lines)
    
    def _text_matches_ivr(self, text: str, ivr_text: str) -> bool:
        """Check if a line of text matches the IVR text."""
        # Normalize both
        text = text.strip()
        ivr_text = ivr_text.strip()
        
        if not text or not ivr_text:
            return False
        
        # Check for substantial overlap
        # If the text is contained in IVR or vice versa
        if text in ivr_text or ivr_text in text:
            return True
        
        # Check word overlap
        text_words = set(text.split())
        ivr_words = set(ivr_text.split())
        
        if not text_words or not ivr_words:
            return False
        
        overlap = len(text_words & ivr_words) / min(len(text_words), len(ivr_words))
        return overlap >= 0.6
    
    def _mark_ivr_words(self, aligned_words: List[Dict], ivr_text: str) -> List[Dict]:
        """Mark words at the beginning as IVR based on ivr_text."""
        if not ivr_text or not aligned_words:
            return aligned_words
        
        # Normalize IVR text
        ivr_text_lower = ivr_text.lower()
        ivr_words_list = ivr_text_lower.split()
        ivr_word_count = len(ivr_words_list)
        
        # Build accumulated text from words to find where IVR ends
        accumulated_text = ""
        ivr_end_idx = -1
        
        for i, word in enumerate(aligned_words):
            word_text = word.get('text', '')
            accumulated_text += word_text.lower() + " "
            
            # Check if we've accumulated the full IVR text
            # Use simple containment check
            if ivr_text_lower in accumulated_text.strip():
                ivr_end_idx = i
                break
            
            # Also check word count as fallback
            word_count = len(accumulated_text.split())
            if word_count >= ivr_word_count:
                # Check overlap
                acc_words = set(accumulated_text.lower().split())
                ivr_words = set(ivr_words_list)
                overlap = len(acc_words & ivr_words) / len(ivr_words)
                if overlap >= 0.8:
                    ivr_end_idx = i
                    break
        
        # Mark words as IVR
        if ivr_end_idx >= 0:
            for i in range(ivr_end_idx + 1):
                aligned_words[i]['original_speaker'] = aligned_words[i].get('speaker', '')
                aligned_words[i]['speaker'] = 'IVR'
            print(f"[LLM Corrector] Marked words [0:{ivr_end_idx}] as IVR")
        
        return aligned_words
    
    def _rebuild_transcription(self, aligned_words: List[Dict]) -> str:
        """Rebuild formatted transcription from aligned words."""
        if not aligned_words:
            return ""
        
        result = []
        current_speaker = None
        current_words = []
        
        for word_info in aligned_words:
            speaker = word_info.get('speaker', 'Unknown')
            text = word_info.get('text', '')
            
            if speaker != current_speaker:
                if current_words and current_speaker:
                    result.append(f"[{current_speaker}] {' '.join(current_words)}")
                current_speaker = speaker
                current_words = [text]
            else:
                current_words.append(text)
        
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
        """Legacy method - calls analyze_transcript internally."""
        corrected_transcript, corrected_words, _ = await self.analyze_transcript(
            labeled_transcription, request_id, aligned_words, sample_size
        )
        return corrected_transcript, corrected_words
    
    async def close(self):
        if self.client:
            await self.client.close()
            self.client = None
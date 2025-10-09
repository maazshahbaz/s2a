/**
 * Enumerations for S2A SDK
 */

export enum JobStatusType {
  PENDING = 'pending',
  PROCESSING = 'processing',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled'
}

export enum IntelligenceMode {
  AUTO_DETECT = 'auto_detect',
  SALES = 'sales',
  SUPPORT = 'support',
  GENERAL = 'general',
  QUICK = 'quick'
}

export enum AudioFormat {
  WAV = 'wav',
  MP3 = 'mp3',
  FLAC = 'flac',
  M4A = 'm4a',
  OGG = 'ogg',
  WEBM = 'webm'
}

export enum Priority {
  LOW = 'low',
  NORMAL = 'normal',
  HIGH = 'high',
  URGENT = 'urgent'
}

export enum CallType {
  SALES_CALL = 'sales_call',
  CUSTOMER_SUPPORT = 'customer_support',
  INTERNAL_MEETING = 'internal_meeting',
  INTERVIEW = 'interview',
  PRESENTATION = 'presentation',
  TRAINING = 'training',
  OTHER = 'other'
}

export enum Intent {
  GENERAL_DISCUSSION = 'general_discussion',
  INFORMATION_GATHERING = 'information_gathering',
  PROBLEM_SOLVING = 'problem_solving',
  DECISION_MAKING = 'decision_making',
  PRODUCT_DEMO = 'product_demo',
  PRICE_NEGOTIATION = 'price_negotiation',
  CONTRACT_DISCUSSION = 'contract_discussion',
  SUPPORT_REQUEST = 'support_request',
  COMPLAINT = 'complaint',
  FOLLOW_UP = 'follow_up',
  INTRODUCTION = 'introduction',
  CLOSING = 'closing'
}

export enum Sentiment {
  VERY_POSITIVE = 'very_positive',
  POSITIVE = 'positive',
  NEUTRAL = 'neutral',
  NEGATIVE = 'negative',
  VERY_NEGATIVE = 'very_negative',
  MIXED = 'mixed'
}
/**
 * S2A JavaScript/TypeScript SDK
 * Official SDK for the S2A Speech-to-Actions Platform
 */

export { S2AClient } from './client';
export * from './types';
export * from './errors';
export {
  JobStatusType,
  IntelligenceMode,
  AudioFormat,
  Priority,
  CallType,
  Intent,
  Sentiment
} from './enums';

// Default export
export { S2AClient as default } from './client';
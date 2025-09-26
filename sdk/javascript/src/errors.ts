/**
 * Error classes for S2A SDK
 */

export class S2AError extends Error {
  statusCode?: number;
  response?: any;

  constructor(message: string, statusCode?: number, response?: any) {
    super(message);
    this.name = 'S2AError';
    this.statusCode = statusCode;
    this.response = response;
  }
}

export class AuthenticationError extends S2AError {
  constructor(message: string = 'Authentication failed') {
    super(message, 401);
    this.name = 'AuthenticationError';
  }
}

export class RateLimitError extends S2AError {
  retryAfter: number;

  constructor(message: string = 'Rate limit exceeded', retryAfter: number = 60) {
    super(message, 429);
    this.name = 'RateLimitError';
    this.retryAfter = retryAfter;
  }
}

export class AudioValidationError extends S2AError {
  constructor(message: string = 'Audio validation failed') {
    super(message, 422);
    this.name = 'AudioValidationError';
  }
}

export class TimeoutError extends S2AError {
  constructor(message: string = 'Request timeout') {
    super(message);
    this.name = 'TimeoutError';
  }
}

export class IntelligenceUnavailableError extends S2AError {
  constructor(message: string = 'Intelligence service temporarily unavailable') {
    super(message, 503);
    this.name = 'IntelligenceUnavailableError';
  }
}
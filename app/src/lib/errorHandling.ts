// app/src/lib/errorHandling.ts
import { ErrorCode } from './errorCodes';

export type AnyError = unknown;

export type NormalizedError = {
  code: string;                // 'ERR_...' or 'ERR_GENERIC'
  meta?: Record<string, any>;  // safe, optional details
  raw: AnyError;               // original error for logging
};

// Type guard to check a string matches one of your codes
export const isErrorCode = (v: any): v is keyof typeof ErrorCode =>
  typeof v === 'string' && v in ErrorCode;

// Extract the engine/Callable code from many shapes of errors
export function extractErrorCode(err: AnyError): string | null {
  const e: any = err;

  // Preferred: Firebase HttpsError.details.error (set by your backend)
  const detailsCode = e?.details?.error;
  if (typeof detailsCode === 'string') return detailsCode;

  // Sometimes the message IS the code (or contains it)
  const msg: string | undefined = typeof e?.message === 'string' ? e.message : undefined;
  if (msg) {
    if (isErrorCode(msg)) return msg;
    const m = msg.match(/(ERR_[A-Z_]+)/);
    if (m) return m[1];
  }

  // Some libs hide info under data or error.response (axios/fetch cases)
  const dataMsg: string | undefined =
    typeof e?.response?.data?.message === 'string' ? e.response.data.message : undefined;
  if (dataMsg) {
    if (isErrorCode(dataMsg)) return dataMsg;
    const m = dataMsg.match(/(ERR_[A-Z_]+)/);
    if (m) return m[1];
  }

  return null;
}

// Normalize to a consistent object your app can handle
export function normalizeError(err: AnyError): NormalizedError {
  const code = extractErrorCode(err) || 'ERR_GENERIC';
  const meta = (err as any)?.details?.meta || (err as any)?.details || undefined;
  return { code, meta, raw: err };
}

// Build the i18n key for the code
export function errorKey(code: string): string {
  return `errors.${code}`;
}

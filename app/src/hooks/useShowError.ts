// app/src/hooks/useShowError.ts
import { useTranslation } from 'react-i18next';
import {normalizeError, errorKey, NormalizedError} from '../lib/errorHandling';

// If you use a toast library, import and use it here.
// Example with window.alert fallback to keep it dependency-free.
// import { toast } from 'react-hot-toast';

export type ShowErrorOptions = {
  fallbackKey?: string; // default generic key
  asToast?: boolean;    // if you plug a toast library
};

export const useShowError = (opts: ShowErrorOptions = {}) => {
  const { t } = useTranslation();
  const { fallbackKey = 'errors.ERR_GENERIC', asToast = false } = opts;

  return (err: NormalizedError) => {
    const code  = err.code; //err must has been previously normalized through app/src/lib/errorHandling.ts normalizeError(err)
    const msg = t(errorKey(code), t(fallbackKey));
    // Swap alert with your toast:
    // asToast ? toast.error(msg) : alert(msg);
    // For now:
    alert(msg);

    // Optional: dev logging
    if (import.meta?.env?.DEV) {
      // eslint-disable-next-line no-console
      console.debug('[showError]', { code, err: raw });
    }
  };
};

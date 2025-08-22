// app/src/lib/callable.ts
import { httpsCallable } from 'firebase/functions';
import { functions } from './firebase';
import { normalizeError } from './errorHandling';

export async function callFn<Req, Res>(name: string, data: Req): Promise<Res> {
  try {
    const fn = httpsCallable<Req, Res>(functions, name);
    const res = await fn(data);
    return res.data as Res;
  } catch (err) {
    throw normalizeError(err); // consumers can rely on {code, meta, raw}
  }
}

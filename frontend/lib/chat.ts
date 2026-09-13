import { ApiError, type Citation, type Role } from "@/lib/api";
import { type TranslationDict } from "@/lib/i18n";

/** A rendered conversation turn (view model over the API types). */
export interface Message {
  role: Role;
  content: string;
  citations?: Citation[];
}

/**
 * Turn any thrown value into a short, user-facing string.
 *
 * Only the "HTTP n" / "network error" prefix is translated — `err.message`
 * itself comes from the backend (FastAPI's error `detail`) or, for a
 * transport failure, from lib/api.ts. Translating arbitrary backend text
 * would need backend-side i18n; out of scope for this UI-only toggle.
 */
export function errorText(err: unknown, t: TranslationDict): string {
  if (err instanceof ApiError) {
    const where = err.status ? `HTTP ${err.status}` : t.networkError;
    return `${where}: ${err.message}`;
  }
  return t.somethingWentWrong;
}

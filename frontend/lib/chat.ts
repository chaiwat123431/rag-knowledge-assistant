import { ApiError, type Citation, type Role } from "@/lib/api";

/** A rendered conversation turn (view model over the API types). */
export interface Message {
  role: Role;
  content: string;
  citations?: Citation[];
}

/** Turn any thrown value into a short, user-facing string. */
export function errorText(err: unknown): string {
  if (err instanceof ApiError) {
    const where = err.status ? `HTTP ${err.status}` : "network error";
    return `${where}: ${err.message}`;
  }
  return "Something went wrong.";
}

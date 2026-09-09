// Network boundary for the RAG backend. The page component calls these
// two functions and never touches `fetch` or response parsing directly.

// `||` not `??`: a blank NEXT_PUBLIC_API_BASE_URL should fall back, not
// become "" (which would make every call a same-origin relative request).
const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

// The local LLM can take a while; the backend's own read timeout is 120s,
// so give queries more headroom. First-ever ingest may download the
// embedding model, so uploads get a generous window too.
const QUERY_TIMEOUT_MS = 150_000;
const UPLOAD_TIMEOUT_MS = 180_000;

export type Role = "user" | "assistant";

export interface HistoryMessage {
  role: Role;
  content: string;
}

export interface Citation {
  marker: number;
  source: string | null;
  chunk_index: number | null;
  text: string;
  score: number | null;
}

export interface QueryResponse {
  answer: string;
  has_context: boolean;
  citations: Citation[];
}

export interface IngestResponse {
  source: string;
  chunks_added: number;
}

/** A backend error with a usable message and the HTTP status. */
export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

// FastAPI puts a string in `detail` for our own HTTPExceptions, but an
// array of {loc, msg, ...} for 422 request-validation errors.
async function readErrorMessage(response: Response): Promise<string> {
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    return response.statusText || `HTTP ${response.status}`;
  }

  const detail = (body as { detail?: unknown })?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        const loc = Array.isArray(item?.loc) ? item.loc.join(".") : "";
        return loc ? `${loc}: ${item?.msg ?? ""}` : String(item?.msg ?? item);
      })
      .join("; ");
  }
  return response.statusText || `HTTP ${response.status}`;
}

async function toApiError(response: Response): Promise<ApiError> {
  return new ApiError(response.status, await readErrorMessage(response));
}

/** fetch() with a timeout, mapping transport/abort failures to ApiError. */
async function request(
  path: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      signal: AbortSignal.timeout(timeoutMs),
    });
  } catch (err) {
    if (
      err instanceof DOMException &&
      (err.name === "TimeoutError" || err.name === "AbortError")
    ) {
      throw new ApiError(0, "The request timed out. Is the backend running?");
    }
    throw new ApiError(0, `Could not reach the API at ${API_BASE}.`);
  }
  if (!response.ok) throw await toApiError(response);
  return response;
}

/** Upload one file to be parsed, chunked and indexed. */
export async function ingestDocument(file: File): Promise<IngestResponse> {
  const form = new FormData();
  form.append("file", file);
  const response = await request(
    "/documents",
    { method: "POST", body: form },
    UPLOAD_TIMEOUT_MS,
  );
  return (await response.json()) as IngestResponse;
}

/** Ask a question, optionally with prior conversation turns. */
export async function askQuestion(
  question: string,
  history: HistoryMessage[],
): Promise<QueryResponse> {
  const response = await request(
    "/query",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history }),
    },
    QUERY_TIMEOUT_MS,
  );
  return (await response.json()) as QueryResponse;
}

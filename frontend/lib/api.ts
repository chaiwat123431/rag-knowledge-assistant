// Network boundary for the RAG backend. The page component calls these
// two functions and never touches `fetch` or response parsing directly.

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

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

/** Upload one file to be parsed, chunked and indexed. */
export async function ingestDocument(file: File): Promise<IngestResponse> {
  const form = new FormData();
  form.append("file", file);

  let response: Response;
  try {
    response = await fetch(`${API_BASE}/documents`, {
      method: "POST",
      body: form,
    });
  } catch {
    throw new ApiError(0, `Could not reach the API at ${API_BASE}.`);
  }

  if (!response.ok) throw await toApiError(response);
  return (await response.json()) as IngestResponse;
}

/** Ask a question, optionally with prior conversation turns. */
export async function askQuestion(
  question: string,
  history: HistoryMessage[],
): Promise<QueryResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history }),
    });
  } catch {
    throw new ApiError(0, `Could not reach the API at ${API_BASE}.`);
  }

  if (!response.ok) throw await toApiError(response);
  return (await response.json()) as QueryResponse;
}

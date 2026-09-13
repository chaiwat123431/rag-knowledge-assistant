// A dictionary-based i18n, not a library: this is one page with a
// handful of short static strings, so a Context + plain object beats
// pulling in next-intl (locale routing, middleware) for no real benefit.
// See LanguageProvider in language-context.tsx for how this is consumed.

export type Language = "en" | "fr";

const en = {
  appTitle: "RAG Assistant",
  newConversation: "New conversation",
  switchLanguage: "Switch language",
  emptyState:
    "No messages yet. Upload a document, then ask something about it.",
  you: "You",
  assistant: "Assistant",
  thinking: "Thinking…",
  thinkingHint: "(the local model can take a few seconds)",
  somethingWentWrong: "Something went wrong.",
  networkError: "network error",
  addDocument: "Add a document",
  upload: "Upload",
  uploading: "Uploading…",
  askPlaceholder: "Ask about your documents…",
  questionLabel: "Question",
  send: "Send",
  unknownSource: "unknown",
  uploadingFile: (name: string) => `Uploading ${name}…`,
  uploadSuccess: (source: string, chunks: number) =>
    `Indexed "${source}" — ${chunks} chunk${chunks === 1 ? "" : "s"} added.`,
  uploadFailed: (detail: string) => `Upload failed — ${detail}`,
  sources: (count: number) => `Sources (${count})`,
  chunkLabel: (index: number) => `chunk ${index}`,
} as const;

// Widen `en`'s literal string/function types to a plain shape (string, or
// a same-signature function) so `fr` only has to match *shape*, not exact
// English text — while still forcing every key to be present with the
// right kind of value. A missing or mistyped translation is a compile
// error, not English text silently left in the French UI.
type Dict = {
  [K in keyof typeof en]: (typeof en)[K] extends (...args: infer A) => string
    ? (...args: A) => string
    : string;
};

const fr: Dict = {
  appTitle: "RAG Assistant",
  newConversation: "Nouvelle conversation",
  switchLanguage: "Changer de langue",
  emptyState:
    "Aucun message pour l'instant. Importez un document, puis posez une question à son sujet.",
  you: "Vous",
  assistant: "Assistant",
  thinking: "Réflexion…",
  thinkingHint: "(le modèle local peut prendre quelques secondes)",
  somethingWentWrong: "Une erreur est survenue.",
  networkError: "erreur réseau",
  addDocument: "Ajouter un document",
  upload: "Importer",
  uploading: "Import en cours…",
  askPlaceholder: "Posez une question sur vos documents…",
  questionLabel: "Question",
  send: "Envoyer",
  unknownSource: "inconnue",
  uploadingFile: (name: string) => `Import de ${name} en cours…`,
  uploadSuccess: (source: string, chunks: number) =>
    `« ${source} » indexé — ${chunks} extrait${chunks === 1 ? "" : "s"} ajouté${chunks === 1 ? "" : "s"}.`,
  uploadFailed: (detail: string) => `Échec de l'import — ${detail}`,
  sources: (count: number) => `Sources (${count})`,
  chunkLabel: (index: number) => `extrait ${index}`,
};

export const translations: Record<Language, Dict> = { en, fr };

export type TranslationDict = Dict;

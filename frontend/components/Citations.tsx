"use client";

import { Card } from "@/components/ui/card";
import { type Citation } from "@/lib/api";
import { useLanguage } from "@/lib/language-context";

export function Citations({ citations }: { citations: Citation[] }) {
  const { t } = useLanguage();

  if (citations.length === 0) return null;

  return (
    <div className="mt-3 flex flex-col gap-2">
      <p className="text-xs font-medium text-muted-foreground">
        {t.sources(citations.length)}
      </p>
      {citations.map((citation) => (
        <Card
          key={citation.marker}
          size="sm"
          className="gap-1.5 ring-border p-3"
        >
          <p className="text-xs font-medium">
            <span className="text-muted-foreground">[{citation.marker}]</span>{" "}
            {citation.source ?? t.unknownSource}
            {citation.chunk_index !== null && (
              <span className="text-muted-foreground">
                {" "}
                · {t.chunkLabel(citation.chunk_index)}
              </span>
            )}
          </p>
          <p className="text-xs whitespace-pre-wrap text-muted-foreground">
            {citation.text}
          </p>
        </Card>
      ))}
    </div>
  );
}

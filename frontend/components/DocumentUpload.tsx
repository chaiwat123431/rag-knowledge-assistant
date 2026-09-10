"use client";

import { useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ingestDocument } from "@/lib/api";
import { errorText } from "@/lib/chat";
import { cn } from "cn";

type UploadState = "idle" | "uploading" | "done" | "error";

export function DocumentUpload() {
  const [file, setFile] = useState<File | null>(null);
  const [state, setState] = useState<UploadState>("idle");
  const [message, setMessage] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  async function handleUpload() {
    if (!file || state === "uploading") return;
    setState("uploading");
    setMessage(`Uploading ${file.name}…`);
    try {
      const res = await ingestDocument(file);
      setState("done");
      setMessage(
        `Indexed “${res.source}” — ${res.chunks_added} chunk${
          res.chunks_added === 1 ? "" : "s"
        } added.`,
      );
      setFile(null);
      if (inputRef.current) inputRef.current.value = "";
    } catch (err) {
      setState("error");
      setMessage(`Upload failed — ${errorText(err)}`);
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Add a document</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center gap-3">
          <input
            ref={inputRef}
            type="file"
            accept=".pdf,.txt,.md"
            onChange={(e) => {
              setFile(e.target.files?.[0] ?? null);
              setState("idle");
              setMessage("");
            }}
            className="text-sm text-muted-foreground file:mr-3 file:rounded-md file:border file:border-border file:bg-background file:px-2.5 file:py-1 file:text-sm file:text-foreground"
          />
          <Button
            type="button"
            variant="outline"
            onClick={handleUpload}
            disabled={!file || state === "uploading"}
          >
            {state === "uploading" ? "Uploading…" : "Upload"}
          </Button>
        </div>
        {message && (
          <p
            className={cn(
              "text-sm",
              state === "error" ? "text-destructive" : "text-muted-foreground",
            )}
          >
            {message}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

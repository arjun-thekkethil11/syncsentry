import { useCallback, useRef, useState } from "react";
import type { ChangeEvent, DragEvent } from "react";

interface FileDropProps {
  label: string;
  hint: string;
  accept: string;
  file: File | null;
  onChange: (file: File | null) => void;
  optional?: boolean;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

export function FileDrop({ label, hint, accept, file, onChange, optional }: FileDropProps) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragOver(false);
      const dropped = e.dataTransfer.files?.[0];
      if (dropped) onChange(dropped);
    },
    [onChange],
  );

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const picked = e.target.files?.[0];
    onChange(picked ?? null);
  };

  return (
    <div>
      <div className="flex items-baseline justify-between mb-2">
        <span className="text-sm font-medium text-slate-200">{label}</span>
        {optional && <span className="text-xs text-slate-500">optional</span>}
      </div>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={`relative cursor-pointer rounded-xl border-2 border-dashed p-6 text-center transition-colors ${
          dragOver
            ? "border-brand-400 bg-brand-500/10"
            : file
              ? "border-emerald-500/40 bg-emerald-500/[0.04]"
              : "border-white/15 hover:border-white/25 bg-white/[0.02]"
        }`}
      >
        <input ref={inputRef} type="file" accept={accept} onChange={handleInputChange} className="hidden" />
        {file ? (
          <div className="flex items-center justify-center gap-3">
            <div className="text-left">
              <p className="text-sm font-medium text-slate-100 truncate max-w-xs">{file.name}</p>
              <p className="text-xs text-slate-500">{formatBytes(file.size)}</p>
            </div>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onChange(null);
                if (inputRef.current) inputRef.current.value = "";
              }}
              className="text-xs text-rose-400 hover:text-rose-300 px-2 py-1 rounded-md hover:bg-rose-500/10"
            >
              remove
            </button>
          </div>
        ) : (
          <>
            <p className="text-sm text-slate-300 mb-1">Drop a file here or click to browse</p>
            <p className="text-xs text-slate-500">{hint}</p>
          </>
        )}
      </div>
    </div>
  );
}

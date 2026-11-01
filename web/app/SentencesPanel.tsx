import { languageName, sentenceViews, type OperationStatus } from "./sentence-operations";

type JsonRecord = Record<string, unknown>;

const statusLabels: Record<OperationStatus, string> = {
  supported: "Supported",
  partial: "Partial",
  unsupported: "Unsupported",
  unknown: "Unknown",
  "rust-only": "Rust only",
};

const statusStyles: Record<OperationStatus, string> = {
  supported: "border-emerald-300 bg-emerald-50 text-emerald-900",
  partial: "border-amber-300 bg-amber-50 text-amber-900",
  unsupported: "border-zinc-300 bg-zinc-100 text-zinc-800",
  unknown: "border-zinc-300 bg-white text-zinc-700",
  "rust-only": "border-sky-300 bg-sky-50 text-sky-900",
};

export function SentencesPanel({ documentReport }: { documentReport: JsonRecord | null }) {
  const sentences = sentenceViews(documentReport);
  return (
    <section>
      <p className="text-xs font-semibold uppercase tracking-[0.12em] text-muted">Per-sentence routing</p>
      <h4 className="mt-1 text-xl font-semibold text-ink">Sentences</h4>
      <p className="mt-1 max-w-3xl text-sm leading-6 text-muted">
        Each sentence is analyzed in its own detected language. Language-detection confidence is the detector&apos;s
        relative score between languages; it is not model or extraction confidence. Operations without a producer for a
        sentence&apos;s language say so instead of borrowing another language&apos;s results, and no relation is drawn across
        sentences.
      </p>
      <div className="mt-4 grid gap-4">
        {sentences.map((sentence) => (
          <article
            key={sentence.index}
            className="rounded-md border border-line bg-surface p-4"
            aria-label={`Sentence ${sentence.index + 1}: ${languageName(sentence.language)}`}
          >
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <p className="text-sm font-semibold text-ink">
                Sentence {sentence.index + 1} · {languageName(sentence.language)}
              </p>
              <p className="text-xs text-muted">UTF-8 bytes {sentence.byteStart}–{sentence.byteEnd}</p>
            </div>
            <blockquote className="mt-2 border-l-2 border-line pl-3 text-base leading-7 text-ink">{sentence.text}</blockquote>
            <p className="mt-2 text-xs text-muted">Language: {sentence.detection}</p>
            <div className="mt-3 overflow-x-auto">
              <table className="w-full min-w-[36rem] border-collapse text-left text-sm">
                <thead>
                  <tr className="border-b border-line text-muted">
                    <th className="py-2 pr-4 font-medium">Operation</th>
                    <th className="py-2 pr-4 font-medium">Status</th>
                    <th className="py-2 font-medium">Producer and evidence</th>
                  </tr>
                </thead>
                <tbody>
                  {sentence.rows.map((row) => (
                    <tr key={row.operation} className="border-b border-line/70 align-top">
                      <th scope="row" className="py-2 pr-4 font-medium text-ink">{row.label}</th>
                      <td className="py-2 pr-4">
                        <span className={`inline-block rounded-full border px-2 py-0.5 text-xs font-semibold ${statusStyles[row.status]}`}>
                          {statusLabels[row.status]}
                        </span>
                      </td>
                      <td className="py-2 text-muted">
                        {row.producer ? <span className="font-mono text-xs text-ink">{row.producer}</span> : null}
                        {row.producer && row.detail ? " · " : null}
                        {row.detail}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

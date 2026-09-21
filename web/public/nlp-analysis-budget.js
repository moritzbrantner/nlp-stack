// Browser execution policy; native capability APIs retain their own contracts.
export const maxAnalysisTextBytes = 256 * 1024;
export const maxAnalysisSentences = 512;

export function assertAnalysisTextBudget(request) {
  const input = request?.input;
  const texts = request?.operation === "analysis.semantic-corpus" && Array.isArray(input?.items)
    ? input.items.map((item) => item?.text)
    : [input?.text];
  let bytes = 0;
  for (const text of texts) {
    if (typeof text !== "string") continue; // Rust owns request validation.
    // Reject huge strings before allocating an encoded copy on the main thread.
    if (text.length > maxAnalysisTextBytes) throw textBudgetError();
    bytes += new TextEncoder().encode(text).byteLength;
    if (bytes > maxAnalysisTextBytes) throw textBudgetError();
  }
}

function textBudgetError() {
  return new Error("Browser analysis supports up to 256 KiB of source text in total. Choose a shorter passage or fewer documents.");
}

export function assertAnalysisSentenceBudget(count) {
  if (count > maxAnalysisSentences) {
    throw new Error("Browser analysis supports up to 512 sentences in total. Choose a shorter passage or fewer documents.");
  }
}

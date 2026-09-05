import { describe, expect, it } from "vitest";

import { extractMarkupText, normalizeExtractedText } from "./browser-text-ingest";

describe("browser text ingestion", () => {
  it("normalizes extracted text without flattening paragraphs", () => {
    expect(normalizeExtractedText("  First   line.\r\n\r\n\r\nSecond\t line.  ")).toBe(
      "First line.\n\nSecond line.",
    );
  });

  it("extracts visible HTML text and drops executable content", () => {
    const text = extractMarkupText(`
      <html><body>
        <h1>Release notes</h1>
        <p>Rust owns the NLP result.</p>
        <script>window.bad = "do not analyze me"</script>
        <style>.hidden { display: none; }</style>
      </body></html>
    `);

    expect(text).toContain("Release notes");
    expect(text).toContain("Rust owns the NLP result.");
    expect(text).not.toContain("do not analyze me");
    expect(text).not.toContain("display: none");
  });
});

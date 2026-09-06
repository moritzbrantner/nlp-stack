import { expect, test } from "playwright/test";

test("keeps single-document semantics separate from multi-document corpus themes", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { level: 1, name: /Analyze text and documents locally with Rust NLP/i })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Try an example" })).toBeVisible();
  await expect(page.getByLabel("Text to analyze")).toBeVisible();
  await expect(page.getByLabel("Upload documents")).toHaveAttribute("multiple", "");

  const dialogue = page.getByRole("button", { name: /Meeting dialogue/ });
  await expect(dialogue).toBeVisible();
  await dialogue.click();
  await expect(page.getByLabel("Text to analyze")).toHaveValue(/Maya:/);

  const analyze = page.getByRole("button", { name: "Analyze text" });
  await expect(analyze).toBeEnabled();
  await analyze.click();

  await expect(page.getByRole("heading", { name: "Semantic map" })).toBeVisible({ timeout: 20_000 });
  await expect(page.getByText(/Source: Meeting dialogue/i)).toBeVisible();

  await page.getByRole("tab", { name: "Corpus themes" }).click();
  await expect(page.getByRole("heading", { name: "Corpus themes" })).toBeVisible();
  await expect(page.getByText(/require at least two supplied documents/i)).toBeVisible();

  await page.getByRole("tab", { name: "Word profile" }).click();
  await expect(page.getByRole("heading", { name: "Word profile" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Ranked terms" })).toBeVisible();

  await page.getByLabel("Upload documents").setInputFiles([
    {
      name: "search-a.txt",
      mimeType: "text/plain",
      buffer: Buffer.from(
        "Semantic search improves retrieval. Vector indexes support semantic search. Tomatoes grow in soil.",
      ),
    },
    {
      name: "search-b.txt",
      mimeType: "text/plain",
      buffer: Buffer.from(
        "Semantic search improves retrieval. Vector indexes support semantic search. Roses grow in soil.",
      ),
    },
  ]);

  await expect(page.getByRole("heading", { name: "Corpus themes" })).toBeVisible({ timeout: 20_000 });
  await expect(page.getByText(/Corpus: 2 sources/i)).toBeVisible();
  await expect(page.getByText(/hashed TF-IDF backend/i)).toBeVisible();
  await expect(page.getByRole("heading", { name: "Theme evidence" })).toBeVisible();
  await expect(page.getByText(/semantic.*retrieval|retrieval.*semantic/i).first()).toBeVisible();
  await expect(page.getByText(/Analysis ready. Corpus themes use 2 supplied sources/i)).toBeVisible();
});

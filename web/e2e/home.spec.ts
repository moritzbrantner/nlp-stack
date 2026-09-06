import { expect, test } from "playwright/test";

test("selects examples and renders corpus views from text-analysis Wasm", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { level: 1, name: /Analyze text and documents locally with Rust NLP/i })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Try an example" })).toBeVisible();
  await expect(page.getByLabel("Text to analyze")).toBeVisible();
  await expect(page.getByLabel("Upload document")).toHaveAttribute("type", "file");

  const dialogue = page.getByRole("button", { name: /Meeting dialogue/ });
  await expect(dialogue).toBeVisible();
  await dialogue.click();
  await expect(page.getByLabel("Text to analyze")).toContainText("Maya:");

  const analyze = page.getByRole("button", { name: "Analyze text" });
  await expect(analyze).toBeEnabled();
  await analyze.click();

  await expect(page.getByRole("heading", { name: "Semantic corpus" })).toBeVisible({ timeout: 20_000 });
  await expect(page.getByRole("heading", { name: "Concept evidence" })).toBeVisible();
  await expect(page.getByText(/Source: Meeting dialogue/i)).toBeVisible();

  await page.getByRole("tab", { name: "Word corpus" }).click();
  await expect(page.getByRole("heading", { name: "Word corpus" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Ranked corpus terms" })).toBeVisible();
  await expect(page.getByText(/Results were produced locally by text-analysis Wasm/i)).toBeVisible();
});

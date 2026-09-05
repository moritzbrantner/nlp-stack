import { expect, test } from "playwright/test";

test("loads and runs the text-analysis Wasm studio", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { level: 1, name: /Analyze text and documents locally with Rust NLP/i })).toBeVisible();
  await expect(page.getByLabel("Text to analyze")).toBeVisible();
  await expect(page.getByLabel("Upload document")).toHaveAttribute("type", "file");

  const analyze = page.getByRole("button", { name: "Analyze text" });
  await expect(analyze).toBeEnabled();
  await analyze.click();

  await expect(page.getByRole("heading", { name: "Extractive summary" })).toBeVisible({ timeout: 20_000 });
  await expect(page.getByText(/Results were produced locally by text-analysis Wasm/i)).toBeVisible();
});

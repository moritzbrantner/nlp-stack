import { expect, test } from "playwright/test";

test("loads the text-core Wasm surface", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { level: 1, name: /Rust NLP package surfaces/i })).toBeVisible();
  await expect(page.getByRole("combobox", { name: "Scenario" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Run" })).toBeEnabled();
});

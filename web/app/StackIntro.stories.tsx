import type { Meta, StoryObj } from "@storybook/nextjs-vite";

import { StackIntro } from "./StackIntro";

const meta = {
  title: "Pages/StackIntro",
  component: StackIntro,
  parameters: {
    layout: "fullscreen",
  },
} satisfies Meta<typeof StackIntro>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

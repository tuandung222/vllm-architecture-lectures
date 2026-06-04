// Trigger build: 2026-06-04 15:15
import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'vLLM Internals',
  tagline: 'Chuỗi bài giảng phân tích chi tiết kiến trúc & hiện thực thư viện vLLM',
  favicon: 'img/favicon.ico',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://tuandung222.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/vllm-architecture-lectures/',

  // GitHub pages deployment config.
  organizationName: 'tuandung222', // GitHub org/user name.
  projectName: 'vllm-architecture-lectures', // Repo name.

  onBrokenLinks: 'warn',

  // Internationalization settings
  i18n: {
    defaultLocale: 'vi',
    locales: ['vi'],
  },

  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  themes: ['@docusaurus/theme-mermaid'],

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          routeBasePath: 'docs',
          editUrl:
            'https://github.com/tuandung222/vllm-architecture-lectures/tree/main/',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: false, // Vô hiệu hóa blog để tập trung vào tài liệu bài học
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css',
      type: 'text/css',
      integrity: 'sha384-GMR9m/t7ic3fIB780ZGydg9s9ID1txv39C21fS9gy047+Fh5hnipf5tL378KvhQx',
      crossorigin: 'anonymous',
    },
  ],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      defaultMode: 'dark',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'vLLM Internals',
      logo: {
        alt: 'vLLM Internals Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Bài học (Lectures)',
        },
        {
          href: 'https://github.com/tuandung222/vllm-architecture-lectures',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Bài học',
          items: [
            {
              label: 'Bài 0: Kiến thức Hệ điều hành',
              to: '/docs/lesson_0_os_fundamentals',
            },
            {
              label: 'Bài 1: Autoregressive & Memory Bottlenecks',
              to: '/docs/lesson_1_memory_bottleneck',
            },
            {
              label: 'Bài 8: Thực hành Serving Engine',
              to: '/docs/lesson_8_toy_serving_engine',
            },
          ],
        },
        {
          title: 'Tài nguyên',
          items: [
            {
              label: 'GitHub Repository',
              href: 'https://github.com/tuandung222/vllm-architecture-lectures',
            },
            {
              label: 'vLLM GitHub',
              href: 'https://github.com/vllm-project/vllm',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} vLLM Internals. Biên soạn bởi tuandung222. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'json', 'yaml', 'markdown'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;

import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'TurboQuant Internals',
  tagline: 'Phân tích chi tiết thuật toán TurboQuant của Google & cách tích hợp vào vLLM KV Cache',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://tuandung222.github.io',
  baseUrl: '/turboquant-architecture-lectures/',

  organizationName: 'tuandung222',
  projectName: 'turboquant-architecture-lectures',

  onBrokenLinks: 'warn',

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
            'https://github.com/tuandung222/turboquant-architecture-lectures/tree/main/',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: false,
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
      integrity: 'sha384-GvrOXuhMATgEsSwCs4smul74iXGOixntILdUW9XmUC6+HX0sLNAK3q71HotJqlAn',
      crossorigin: 'anonymous',
    },
  ],

  themeConfig: {
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      defaultMode: 'dark',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'TurboQuant Internals',
      logo: {
        alt: 'TurboQuant Internals Logo',
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
          href: 'https://arxiv.org/abs/2504.19874',
          label: 'Paper (arXiv)',
          position: 'right',
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
              label: 'Roadmap & Syllabus',
              to: '/docs/roadmap',
            },
            {
              label: 'Bài 0: Nền tảng Vector Quantization',
              to: '/docs/lesson_0_vector_quantization_background',
            },
            {
              label: 'Bài 5: Tích hợp vào vLLM',
              to: '/docs/lesson_5_vllm_integration',
            },
          ],
        },
        {
          title: 'Tài nguyên',
          items: [
            {
              label: 'TurboQuant Paper (arXiv 2504.19874)',
              href: 'https://arxiv.org/abs/2504.19874',
            },
            {
              label: 'vLLM GitHub',
              href: 'https://github.com/vllm-project/vllm',
            },
            {
              label: 'vLLM Internals Lectures',
              href: 'https://github.com/tuandung222/vllm-architecture-lectures',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} TurboQuant Internals. Biên soạn bởi tuandung222. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'json', 'yaml', 'markdown'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;

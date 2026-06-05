import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero custom-hero', styles.heroBanner)}>
      <div className="container" style={{ position: 'relative', zIndex: 1 }}>
        <div style={{ display: 'inline-flex', padding: '4px 12px', background: 'rgba(13, 148, 136, 0.15)', color: '#2dd4bf', borderRadius: '100px', fontSize: '0.85rem', fontWeight: 600, border: '1px solid rgba(13, 148, 136, 0.25)', marginBottom: '1.5rem' }}>
          GOOGLE RESEARCH · ICLR 2026 · arXiv 2504.19874
        </div>
        <Heading as="h1" className="hero__title" style={{ fontSize: '3.5rem', fontWeight: 800, background: 'linear-gradient(135deg, #fff 30%, #2dd4bf 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', marginBottom: '1.5rem' }}>
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle" style={{ maxWidth: '780px', margin: '0 auto 2.5rem auto', fontSize: '1.2rem', lineHeight: '1.6', opacity: 0.85 }}>
          {siteConfig.tagline}
        </p>
        <div className={styles.buttons}>
          <Link
            className="button button--primary button--lg"
            style={{ padding: '0.8rem 2rem', fontSize: '1.05rem', fontWeight: 600, borderRadius: '8px', boxShadow: '0 4px 14px rgba(13, 148, 136, 0.4)', transition: 'all 0.3s ease' }}
            to="/docs/roadmap">
            Bắt đầu học ngay 🚀
          </Link>
          <Link
            className="button button--outline button--secondary button--lg"
            style={{ padding: '0.8rem 2rem', fontSize: '1.05rem', fontWeight: 600, borderRadius: '8px', marginLeft: '1rem', border: '1px solid var(--card-border)' }}
            href="https://arxiv.org/abs/2504.19874">
            Đọc Paper gốc 📄
          </Link>
        </div>
      </div>
    </header>
  );
}

interface FeatureItem {
  title: string;
  badge: string;
  description: string;
}

const CorePillars: FeatureItem[] = [
  {
    title: 'Random Rotation',
    badge: 'Data-Oblivious',
    description: 'Xoay ngẫu nhiên vector (qua randomized Hadamard transform) khiến mọi tọa độ tuân theo cùng một phân phối Beta đã biết và gần như độc lập — không cần dữ liệu hiệu chỉnh (calibration-free), chạy online.',
  },
  {
    title: 'Per-Coordinate Quantizer',
    badge: 'MSE Near-Optimal',
    description: 'Vì sau khi xoay mọi tọa độ cùng phân phối, chỉ cần thiết kế MỘT bộ lượng hóa vô hướng tối ưu (Lloyd-Max) áp cho từng tọa độ, đạt sai số MSE gần với cận dưới lý thuyết.',
  },
  {
    title: 'QJL Residual',
    badge: 'Unbiased Inner Product',
    description: 'Bộ lượng hóa MSE gây thiên lệch (bias) khi ước lượng tích vô hướng. TurboQuant thêm 1-bit Quantized JL trên phần dư (residual) để thu được ước lượng inner product KHÔNG thiên lệch.',
  },
];

interface LectureItem {
  number: string;
  title: string;
  desc: string;
  path: string;
  category: 'Background' | 'Core Theory' | 'Theory Deep Dive' | 'Integration' | 'Application' | 'Practice';
}

const Lectures: LectureItem[] = [
  {
    number: 'Bài 0',
    title: 'Nền tảng Vector Quantization & Rate-Distortion',
    desc: 'Lý thuyết mã hóa nguồn của Shannon, scalar vs vector quantization, hai loại méo (MSE distortion vs inner product distortion) và quan hệ rate-distortion 6 dB/bit.',
    path: '/docs/lesson_0_vector_quantization_background',
    category: 'Background'
  },
  {
    number: 'Bài 1',
    title: 'Bài toán nén KV Cache & yêu cầu Data-Oblivious',
    desc: 'Vì sao KV Cache là điểm nghẽn VRAM của LLM serving? Phân biệt PTQ tĩnh (calibration) vs lượng hóa online/data-oblivious — yêu cầu cốt lõi mà TurboQuant giải quyết.',
    path: '/docs/lesson_1_kv_cache_problem',
    category: 'Core Theory'
  },
  {
    number: 'Bài 2',
    title: 'Trụ cột 1: Random Rotation & Phân phối Beta',
    desc: 'Toán học của phép xoay ngẫu nhiên: vì sao tọa độ của vector đơn vị sau khi xoay tuân theo phân phối Beta, tính gần-độc-lập ở chiều cao, và randomized Hadamard transform O(d log d).',
    path: '/docs/lesson_2_random_rotation_beta',
    category: 'Core Theory'
  },
  {
    number: 'Bài 3',
    title: 'Trụ cột 2: MSE Scalar Quantizer tối ưu',
    desc: 'Thiết kế bộ lượng hóa vô hướng Lloyd-Max tối ưu cho phân phối tọa độ, công thức méo D(b) ∝ 2^(−2b), vì sao "chia để trị" theo từng tọa độ lại gần tối ưu.',
    path: '/docs/lesson_3_mse_scalar_quantizer',
    category: 'Core Theory'
  },
  {
    number: 'Bài 4',
    title: 'Trụ cột 3: Inner Product & QJL Unbiased',
    desc: 'Vì sao quantizer tối ưu MSE lại thiên lệch khi ước lượng tích vô hướng (attention scores), và cách tiếp cận hai pha: MSE quantizer + 1-bit Quantized JL trên residual.',
    path: '/docs/lesson_4_inner_product_qjl',
    category: 'Core Theory'
  },
  {
    number: 'Bài 5',
    title: 'Tích hợp TurboQuant vào vLLM KV Cache',
    desc: 'Ánh xạ thuật toán vào kiến trúc vLLM: PagedAttention blocks, attention backend, online quant trong vòng lặp decode, so sánh với FP8 KV Cache có sẵn và những điểm cần custom kernel.',
    path: '/docs/lesson_5_vllm_integration',
    category: 'Integration'
  },
  {
    number: 'Bài 6',
    title: 'Cận dưới lý thuyết & Tính tối ưu (~2.7)',
    desc: 'Chứng minh cận dưới information-theoretic cho mọi vector quantizer, vì sao TurboQuant chỉ cách cận này một hằng số nhỏ (~2.7) đồng đều ở mọi bit-width và mọi chiều.',
    path: '/docs/lesson_6_lower_bound_optimality',
    category: 'Theory Deep Dive'
  },
  {
    number: 'Bài 7',
    title: 'Ứng dụng: Nearest Neighbor Search & Vector DB',
    desc: 'Dùng TurboQuant thay cho Product Quantization trong tìm kiếm lân cận gần nhất: recall cao hơn, thời gian indexing gần như bằng 0 (0.0013s vs 239s của PQ).',
    path: '/docs/lesson_7_nearest_neighbor_search',
    category: 'Application'
  },
  {
    number: 'Bài 8',
    title: 'Thực hành: Tự xây dựng Toy TurboQuant',
    desc: 'Lập trình từ con số 0 bằng Python & NumPy: random rotation, MSE quantizer, QJL residual, mô phỏng nén KV Cache và đo MSE/recall để kiểm chứng lý thuyết.',
    path: '/docs/lesson_8_toy_turboquant',
    category: 'Practice'
  }
];

function CategoryBadge({ category }: { category: LectureItem['category'] }) {
  const colors: Record<LectureItem['category'], { bg: string, text: string }> = {
    'Background': { bg: 'rgba(59, 130, 246, 0.15)', text: '#60a5fa' },
    'Core Theory': { bg: 'rgba(16, 185, 129, 0.15)', text: '#34d399' },
    'Theory Deep Dive': { bg: 'rgba(245, 158, 11, 0.15)', text: '#fbbf24' },
    'Integration': { bg: 'rgba(13, 148, 136, 0.18)', text: '#2dd4bf' },
    'Application': { bg: 'rgba(236, 72, 153, 0.15)', text: '#f472b6' },
    'Practice': { bg: 'rgba(99, 102, 241, 0.15)', text: '#818cf8' },
  };

  const color = colors[category];

  return (
    <span style={{ display: 'inline-block', padding: '2px 8px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 600, background: color.bg, color: color.text, alignSelf: 'flex-start' }}>
      {category}
    </span>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title} | Phân tích thuật toán & tích hợp vLLM`}
      description="Chuỗi bài giảng tiếng Việt phân tích chi tiết thuật toán TurboQuant của Google (Online Vector Quantization with Near-optimal Distortion Rate) và cách tích hợp vào vLLM KV Cache.">
      <HomepageHeader />

      <main style={{ padding: '4rem 0', background: 'var(--ifm-background-color)' }}>
        {/* Core Pillars Section */}
        <section className="container" style={{ marginBottom: '5rem' }}>
          <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
            <Heading as="h2" style={{ fontSize: '2rem', fontWeight: 700 }}>
              ⚡ Ba Trụ Cột Của TurboQuant
            </Heading>
            <p style={{ opacity: 0.7, maxWidth: '640px', margin: '0.5rem auto 0 auto' }}>
              Một thuật toán lượng hóa vector data-oblivious, gần tối ưu về méo, chạy online — không cần calibration.
            </p>
          </div>

          <div className="row">
            {CorePillars.map((item, idx) => (
              <div key={idx} className="col col--4" style={{ marginBottom: '1.5rem' }}>
                <div className="glass-panel" style={{ padding: '2rem', height: '100%', display: 'flex', flexDirection: 'column' }}>
                  <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--ifm-color-primary)', textTransform: 'uppercase', marginBottom: '0.5rem', display: 'block' }}>
                    {item.badge}
                  </span>
                  <Heading as="h3" style={{ fontSize: '1.35rem', marginBottom: '1rem', fontWeight: 600 }}>
                    {item.title}
                  </Heading>
                  <p style={{ opacity: 0.8, fontSize: '0.95rem', lineHeight: '1.6', margin: 0 }}>
                    {item.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Lectures Curriculum Section */}
        <section className="container">
          <div style={{ textAlign: 'center', marginBottom: '3.5rem' }}>
            <Heading as="h2" style={{ fontSize: '2rem', fontWeight: 700 }}>
              📚 Giáo Trình Học Tập (Curriculum)
            </Heading>
            <p style={{ opacity: 0.7, maxWidth: '640px', margin: '0.5rem auto 0 auto' }}>
              Đi từ lý thuyết rate-distortion, qua ba trụ cột thuật toán, tới tích hợp thực tế vào vLLM và tự code lại từ đầu.
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '1.5rem' }}>
            {Lectures.map((lecture, idx) => (
              <Link
                to={lecture.path}
                key={idx}
                style={{ textDecoration: 'none', color: 'inherit' }}
              >
                <div className="glass-panel" style={{ padding: '1.5rem', height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', cursor: 'pointer' }}>
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                      <span style={{ fontSize: '0.9rem', fontWeight: 800, color: 'var(--ifm-color-primary)' }}>
                        {lecture.number}
                      </span>
                      <CategoryBadge category={lecture.category} />
                    </div>
                    <Heading as="h3" style={{ fontSize: '1.15rem', marginBottom: '0.75rem', fontWeight: 600, lineHeight: '1.4' }}>
                      {lecture.title}
                    </Heading>
                    <p style={{ opacity: 0.8, fontSize: '0.9rem', lineHeight: '1.5', margin: 0 }}>
                      {lecture.desc}
                    </p>
                  </div>
                  <div style={{ marginTop: '1.5rem', display: 'flex', alignItems: 'center', fontSize: '0.85rem', fontWeight: 600, color: 'var(--ifm-color-primary-light)' }}>
                    Đọc bài học này <span style={{ marginLeft: '4px', transition: 'transform 0.2s ease' }} className="arrow-icon">→</span>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </section>

        {/* Practice Engine Banner */}
        <section className="container" style={{ marginTop: '6rem' }}>
          <div className="glass-panel" style={{ padding: '3rem', background: 'radial-gradient(circle at 90% 10%, rgba(13, 148, 136, 0.12) 0%, transparent 60%), var(--card-bg)', textAlign: 'center', borderRadius: '16px' }}>
            <Heading as="h2" style={{ fontSize: '1.8rem', fontWeight: 700, marginBottom: '1rem' }}>
              💻 Thực Hành: Tự Code "Toy TurboQuant"
            </Heading>
            <p style={{ maxWidth: '720px', margin: '0 auto 2rem auto', opacity: 0.8, lineHeight: '1.6' }}>
              Không gì giúp hiểu bản chất tốt bằng tự lập trình! Trong Bài 8, chúng ta sẽ hiện thực bằng Python &amp; NumPy toàn bộ đường ống TurboQuant — random rotation, bộ lượng hóa MSE, QJL residual — rồi mô phỏng nén KV Cache và đo đạc sai số MSE cùng độ chính xác tích vô hướng.
            </p>
            <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem', flexWrap: 'wrap' }}>
              <Link
                className="button button--primary button--lg"
                style={{ borderRadius: '8px', padding: '0.7rem 1.8rem', fontWeight: 600 }}
                to="/docs/lesson_8_toy_turboquant">
                Đến Bài Học Thực Hành
              </Link>
              <Link
                className="button button--outline button--secondary button--lg"
                style={{ borderRadius: '8px', padding: '0.7rem 1.8rem', fontWeight: 600, border: '1px solid var(--card-border)' }}
                href="https://github.com/tuandung222/vllm-architecture-lectures/tree/main/turboquant-lectures/toy_quant">
                Xem Thư Mục Code
              </Link>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}

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
        <div style={{ display: 'inline-flex', padding: '4px 12px', background: 'rgba(139, 92, 246, 0.15)', color: '#a78bfa', borderRadius: '100px', fontSize: '0.85rem', fontWeight: 600, border: '1px solid rgba(139, 92, 246, 0.25)', marginBottom: '1.5rem' }}>
          DOCUMENTATION & PRACTICAL COURSE
        </div>
        <Heading as="h1" className="hero__title" style={{ fontSize: '3.5rem', fontWeight: 800, background: 'linear-gradient(135deg, #fff 30%, #a78bfa 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', marginBottom: '1.5rem' }}>
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle" style={{ maxWidth: '750px', margin: '0 auto 2.5rem auto', fontSize: '1.2rem', lineHeight: '1.6', opacity: 0.85 }}>
          {siteConfig.tagline}
        </p>
        <div className={styles.buttons}>
          <Link
            className="button button--primary button--lg"
            style={{ padding: '0.8rem 2rem', fontSize: '1.05rem', fontWeight: 600, borderRadius: '8px', boxShadow: '0 4px 14px rgba(139, 92, 246, 0.4)', transition: 'all 0.3s ease' }}
            to="/docs/roadmap">
            Bắt đầu học ngay 🚀
          </Link>
          <Link
            className="button button--outline button--secondary button--lg"
            style={{ padding: '0.8rem 2rem', fontSize: '1.05rem', fontWeight: 600, borderRadius: '8px', marginLeft: '1rem', border: '1px solid var(--card-border)' }}
            href="https://github.com/tuandung222/vllm-architecture-lectures">
            View on GitHub 🛠️
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
    title: 'PagedAttention',
    badge: 'Memory Management',
    description: 'Ứng dụng nguyên lý bộ nhớ ảo (OS Virtual Paging) để loại bỏ phân mảnh bộ nhớ ngoài, giảm thiểu tối đa hao phí KV Cache trong VRAM GPU.',
  },
  {
    title: 'Continuous Batching',
    badge: 'Scheduling',
    description: 'Cơ chế lập lịch ở mức độ token (Iteration-level Scheduling) giúp chèn các yêu cầu prefill và decode động, nâng cao throughput của GPU.',
  },
  {
    title: 'Decoupled Engine (v1)',
    badge: 'Architecture',
    description: 'Mô hình đa tiến trình tách biệt API Server (FastAPI) khỏi phần lõi tính toán GPU (EngineCore), giao tiếp phi trạng thái qua ZeroMQ IPC/Shared Memory.',
  },
];

interface LectureItem {
  number: string;
  title: string;
  desc: string;
  path: string;
  category: 'Background' | 'Core Theory' | 'Deep Dive Code' | 'Optimization' | 'Practice';
}

const Lectures: LectureItem[] = [
  {
    number: 'Bài 0',
    title: 'Kiến thức Hệ điều hành bổ trợ',
    desc: 'Hiểu cơ chế phân trang bộ nhớ (Paging, Page Table), giao tiếp đa tiến trình (ZMQ, Shared Memory) dành cho AI Serving.',
    path: '/docs/lesson_0_os_fundamentals',
    category: 'Background'
  },
  {
    number: 'Bài 0.1',
    title: 'Cấu tạo Phần cứng GPU & Bản chất Prefill/Decode',
    desc: 'Bản chất vật lý của GPU (SMs, HBM, SRAM), tính toán GEMM (Compute-bound) vs nạp weight (Memory-bound) và mô hình Roofline.',
    path: '/docs/lesson_0_gpu_hardware',
    category: 'Background'
  },
  {
    number: 'Bài 1',
    title: 'Autoregressive Serving & Memory Bottlenecks',
    desc: 'Tại sao LLM serving bị nghẽn băng thông bộ nhớ? Công thức tính toán KV Cache thực tế và thách thức phân mảnh bộ nhớ.',
    path: '/docs/lesson_1_memory_bottleneck',
    category: 'Core Theory'
  },
  {
    number: 'Bài 1.1',
    title: 'Chuyển dịch Decode sang Compute-bound',
    desc: 'Chứng minh toán học về sự thay đổi của Arithmetic Intensity theo Batch Size, cơ chế khấu hao trọng số trên SRAM và giới hạn phần cứng.',
    path: '/docs/lesson_1_1_batch_size_compute_bound',
    category: 'Core Theory'
  },
  {
    number: 'Bài 2',
    title: 'PagedAttention & Block Allocation',
    desc: 'Thiết kế PagedAttention, cơ chế Copy-on-Write (CoW) trong Parallel Sampling và cách ánh xạ logical sang physical blocks.',
    path: '/docs/lesson_2_paged_attention',
    category: 'Core Theory'
  },
  {
    number: 'Bài 2.1',
    title: 'Phân tách Kiến trúc Attention Backends',
    desc: 'Tách biệt Memory Layout (Paged KV Cache) khỏi Compute Kernels. Phân tích FlashAttention-2, FlashInfer, Triton, và FlexAttention.',
    path: '/docs/lesson_2_1_attention_backends',
    category: 'Core Theory'
  },
  {
    number: 'Bài 2.2',
    title: 'RadixAttention & Prefix Caching',
    desc: 'Cơ chế Automatic Prefix Caching (APC) qua cây tiền tố Radix Tree, thuật toán LRU Eviction giải phóng block khi cạn bộ nhớ GPU.',
    path: '/docs/lesson_2_2_prefix_caching',
    category: 'Core Theory'
  },
  {
    number: 'Bài 3',
    title: 'Continuous Batching & Preemption',
    desc: 'Giải thuật Continuous Batching, giải pháp thu hồi bộ nhớ Recomputation vs Swapping, kỹ thuật Chunked Prefill giảm giật cục độ trễ.',
    path: '/docs/lesson_3_continuous_batching',
    category: 'Core Theory'
  },
  {
    number: 'Bài 3.2',
    title: 'Chunked Prefill & Mixed Batching',
    desc: 'Bẻ nhỏ các prompt prefill siêu dài thành các chunk kích thước cố định chạy đan xen với decode, loại bỏ hoàn toàn hiện tượng nghẽn.',
    path: '/docs/lesson_3_2_chunked_prefill',
    category: 'Core Theory'
  },
  {
    number: 'Bài 4',
    title: 'Async Serving, Concurrency & Streaming',
    desc: 'Lập trình async, cơ chế ZMQ Router-Dealer, Server-Sent Events (SSE) để stream token và hủy bỏ request (Abort) khi ngắt kết nối.',
    path: '/docs/lesson_4_async_concurrency',
    category: 'Core Theory'
  },
  {
    number: 'Bài 4.1',
    title: 'SHM & ZeroMQ IPC trong Decoupled Engine',
    desc: 'Phân tách tiến trình API (FastAPI) và GPU Engine. Giao tiếp qua ZeroMQ Router-Dealer và truyền tải Zero-copy qua Shared Memory.',
    path: '/docs/lesson_4_1_shared_memory_ipc',
    category: 'Core Theory'
  },
  {
    number: 'Bài 5',
    title: 'Codebase Deep Dive: Scheduler & Block Manager',
    desc: 'Khảo sát mã nguồn vLLM v1: RequestQueue, Scheduler (budget token), KVCacheManager (quản lý bảng trang) và EngineCore.',
    path: '/docs/lesson_5_scheduler_code',
    category: 'Deep Dive Code'
  },
  {
    number: 'Bài 6',
    title: 'Codebase Deep Dive: Distributed Executor & GPU Workers',
    desc: 'Phân tích cơ chế suy luận phân tán (TP/PP) qua NCCL, quy trình đo đạc VRAM (Memory Profiling) và tối ưu hóa qua CUDA Graphs.',
    path: '/docs/lesson_6_distributed_worker',
    category: 'Deep Dive Code'
  },
  {
    number: 'Bài 6.2',
    title: 'Song song Tensor & Giao tiếp NCCL',
    desc: 'Mổ xẻ toán học chia ma trận Column/Row Parallel của Megatron-LM, giải thuật Ring-based All-Reduce truyền thông GPU liên kết NVLink.',
    path: '/docs/lesson_6_2_distributed_comm_nccl',
    category: 'Deep Dive Code'
  },
  {
    number: 'Bài 6.3',
    title: 'CUDA Graph & Shape Bucketing',
    desc: 'Giảm thiểu CPU launch overhead bằng CUDA Graphs, kỹ thuật đắp đệm Shape Bucketing để phục vụ batch size biến đổi linh động.',
    path: '/docs/lesson_6_3_cuda_graph_bucketing',
    category: 'Deep Dive Code'
  },
  {
    number: 'Bài 6.4',
    title: 'Ray vs Multiprocessing: Điều phối Worker & Xử lý lỗi',
    desc: 'Bản chất Ray actor overhead, kiến trúc MultiprocExecutor cục bộ qua ZeroMQ/Shared Memory, và cơ chế phát hiện crash GPU OOM.',
    path: '/docs/lesson_6_4_ray_multiprocessing_orchestration',
    category: 'Deep Dive Code'
  },
  {
    number: 'Bài 6.5',
    title: 'Phục vụ MoE trên Multi-GPU: Expert Parallelism & EPLB',
    desc: 'Giao tiếp All-to-All trong EP routing, lệch tải chuyên gia (expert imbalance), và giải thuật cân bằng expert bằng EPLB.',
    path: '/docs/lesson_6_5_moe_expert_parallelism_eplb',
    category: 'Deep Dive Code'
  },
  {
    number: 'Bài 6.6',
    title: 'NCCL Bottlenecks & Tối ưu hóa mạng phục vụ',
    desc: 'Lý do All-Reduce nghẽn ở decode, so sánh vật lý NVLink vs PCIe, và cẩm nang tinh chỉnh biến môi trường NCCL thực chiến.',
    path: '/docs/lesson_6_6_nccl_bottlenecks_networking',
    category: 'Optimization'
  },
  {
    number: 'Bài 6.7',
    title: 'Context Parallelism & Ring Attention',
    desc: 'Sequence dimension sharding qua cp_utils.py cho chuỗi 1M+ tokens, và cơ chế xoay vòng KV tensor qua giải thuật Ring Attention.',
    path: '/docs/lesson_6_7_context_parallelism_ring_attention',
    category: 'Deep Dive Code'
  },
  {
    number: 'Bài 6.8',
    title: 'Data Parallelism (DP) Replicas & API Routing',
    desc: 'Nhân bản serving replicas để tăng throughput, các giải thuật load balancing và cơ chế Prefix-Aware Routing chống phân mảnh cache.',
    path: '/docs/lesson_6_8_data_parallelism_replicas_routing',
    category: 'Optimization'
  },
  {
    number: 'Bài 7',
    title: 'Kỹ thuật Tối ưu hóa Nâng cao',
    desc: 'Phân tích Speculative Decoding (giải mã suy đoán), nạp động Multi-LoRA (Punica/SGMV kernels) và lượng hóa mô hình.',
    path: '/docs/lesson_7_advanced_serving',
    category: 'Optimization'
  },
  {
    number: 'Bài 7.1',
    title: 'Chi tiết Lượng hóa: AWQ vs Activation Quantization',
    desc: 'So sánh chi tiết lượng hóa Weight-Only (AWQ) bảo vệ salient weights vs lượng hóa Activation và FP8/W8A8/KV Cache.',
    path: '/docs/lesson_7_quantization_deep_dive',
    category: 'Optimization'
  },
  {
    number: 'Bài 7.2',
    title: 'Toán học & Cache trong Speculative Decoding',
    desc: 'Chứng minh giải thuật Rejection Sampling kiểm chứng token nháp, luồng quản lý và rollback KV Cache tức thì ở Block Manager.',
    path: '/docs/lesson_7_2_speculative_decoding_deep_dive',
    category: 'Optimization'
  },
  {
    number: 'Bài 7.2.1',
    title: 'Đi sâu mã nguồn: Cách vLLM hiện thực hóa Speculative Decoding',
    desc: 'Thiết kế Worker v0 vs v1, luồng Propose-Verify trong model_runner.py, và chi tiết RejectionSampler Triton kernels.',
    path: '/docs/lesson_7_2_1_speculative_decoding_vllm_impl',
    category: 'Deep Dive Code'
  },
  {
    number: 'Bài 7.2.2',
    title: 'Triển khai Speculative Decoding trên Production',
    desc: 'So sánh các phương pháp (Medusa, EAGLE, MTP, Ngram), tham số cấu hình CLI, cạm bẫy TP mismatch, và giám sát Acceptance Rate.',
    path: '/docs/lesson_7_2_2_speculative_decoding_production',
    category: 'Optimization'
  },
  {
    number: 'Bài 7.3',
    title: 'Multi-LoRA Serving & SGMV/Punica',
    desc: 'Phục vụ hàng ngàn LoRA adapters đồng thời bằng custom kernels BGMV (cho decode) và SGMV (cho prefill), quản lý cache swap qua PCIe.',
    path: '/docs/lesson_7_3_multi_lora_serving',
    category: 'Optimization'
  },
  {
    number: 'Bài 7.4',
    title: 'Phân rã Prefill & Decode',
    desc: 'Kiến trúc disaggregated serving tách riêng node prefill (compute-bound) và node decode (memory-bound), truyền KV Cache qua GPUDirect RDMA.',
    path: '/docs/lesson_7_4_prefill_decode_disaggregation',
    category: 'Optimization'
  },
  {
    number: 'Bài 8',
    title: 'Thiết kế & Hiện thực Toy Serving Engine',
    desc: 'Tự tay lập trình một Serving Engine tối giản bằng Python: Page allocator, Continuous Scheduler, FastAPI SSE Streaming và Abort.',
    path: '/docs/lesson_8_toy_serving_engine',
    category: 'Practice'
  }
];

function CategoryBadge({ category }: { category: LectureItem['category'] }) {
  const colors: Record<LectureItem['category'], { bg: string, text: string }> = {
    'Background': { bg: 'rgba(59, 130, 246, 0.15)', text: '#60a5fa' },
    'Core Theory': { bg: 'rgba(16, 185, 129, 0.15)', text: '#34d399' },
    'Deep Dive Code': { bg: 'rgba(245, 158, 11, 0.15)', text: '#fbbf24' },
    'Optimization': { bg: 'rgba(236, 72, 153, 0.15)', text: '#f472b6' },
    'Practice': { bg: 'rgba(139, 92, 246, 0.15)', text: '#a78bfa' },
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
      title={`${siteConfig.title} | Deep Dive LLM Serving Architecture`}
      description="Chuỗi bài giảng phân tích chi tiết kiến trúc, thuật toán và mã nguồn của thư viện vLLM dành cho AI Serving & Deep Learning Engineers.">
      <HomepageHeader />
      
      <main style={{ padding: '4rem 0', background: 'var(--ifm-background-color)' }}>
        {/* Core Pillars Section */}
        <section className="container" style={{ marginBottom: '5rem' }}>
          <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
            <Heading as="h2" style={{ fontSize: '2rem', fontWeight: 700 }}>
              🚀 Ba Trụ Cột Tối Ưu Của vLLM
            </Heading>
            <p style={{ opacity: 0.7, maxWidth: '600px', margin: '0.5rem auto 0 auto' }}>
              Những đột phá công nghệ giúp vLLM dẫn đầu về tốc độ và hiệu suất phục vụ LLM
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
            <p style={{ opacity: 0.7, maxWidth: '600px', margin: '0.5rem auto 0 auto' }}>
              Đi từ kiến thức nền tảng hệ điều hành, cấu trúc GPU cho tới chi tiết mã nguồn và thực hành tự code engine.
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
          <div className="glass-panel" style={{ padding: '3rem', background: 'radial-gradient(circle at 90% 10%, rgba(139, 92, 246, 0.12) 0%, transparent 60%), var(--card-bg)', textAlign: 'center', borderRadius: '16px' }}>
            <Heading as="h2" style={{ fontSize: '1.8rem', fontWeight: 700, marginBottom: '1rem' }}>
              💻 Thực Hành: Tự Xây Dựng "Toy Serving Engine"
            </Heading>
            <p style={{ maxWidth: '700px', margin: '0 auto 2rem auto', opacity: 0.8, lineHeight: '1.6' }}>
              Không gì giúp hiểu bản chất tốt bằng tự lập trình! Trong Bài 8, chúng ta sẽ tự tay triển khai bằng Python một máy chủ suy luận LLM tối giản hoàn chỉnh từ con số 0 với Paged Block Allocator, Continuous Batching Scheduler, và FastAPI Async Stream Server.
            </p>
            <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem', flexWrap: 'wrap' }}>
              <Link
                className="button button--primary button--lg"
                style={{ borderRadius: '8px', padding: '0.7rem 1.8rem', fontWeight: 600 }}
                to="/docs/lesson_8_toy_serving_engine">
                Đến Bài Học Thực Hành
              </Link>
              <Link
                className="button button--outline button--secondary button--lg"
                style={{ borderRadius: '8px', padding: '0.7rem 1.8rem', fontWeight: 600, border: '1px solid var(--card-border)' }}
                href="https://github.com/tuandung222/vllm-architecture-lectures/tree/main/toy_engine">
                Xem Thư Mục Code
              </Link>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}

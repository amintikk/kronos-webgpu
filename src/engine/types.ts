/** One OHLCV candle. `t` is the open time in ms (UTC); `a` is the quote-asset volume ("amount"). */
export interface Candle {
  t: number;
  o: number;
  h: number;
  l: number;
  c: number;
  v: number;
  a: number;
}

export interface FileMeta {
  file: string;
  bytes: number;
  sha256: string;
}

export interface KronosMeta {
  label: string;
  params: string;
  source: string;
  tokenizer: string;
  max_context: number;
  n_layers: number;
  n_heads: number;
  d_model: number;
  head_dim: number;
  cross_heads: number;
  cross_head_dim: number;
  s1_vocab: number;
  s2_vocab: number;
  lm: FileMeta;
  s2: FileMeta;
}

export interface TokenizerMeta {
  source: string;
  s1_bits: number;
  s2_bits: number;
  d_in: number;
  enc: FileMeta;
  dec: FileMeta;
}

export interface ChronosMeta {
  label: string;
  params: string;
  source: string;
  context_length: number;
  patch: number;
  prediction_length: number;
  quantiles: number[];
  model: FileMeta;
}

export interface Manifest {
  models: Record<string, KronosMeta>;
  tokenizers: Record<string, TokenizerMeta>;
  chronos: Record<string, ChronosMeta>;
}

export interface KronosParams {
  predLen: number;
  samples: number;
  /** Samples generated in parallel per run. */
  batch: number;
  temperature: number;
  topK: number;
  topP: number;
  seed: number;
  clip?: number;
  /**
   * When context + horizon exceed the model window: false (default) slides the window one candle per
   * step like the original Kronos, rebuilding the cache every step; true drops up to 64 candles at once
   * and rebuilds rarely, which is much faster but lets the model see slightly less history.
   */
  fastWindow?: boolean;
}

/** Minimal surface of the onnxruntime API shared by onnxruntime-web and onnxruntime-node. */
export interface OrtTensor {
  data: unknown;
  dims: readonly number[];
  location?: string;
  dispose(): void;
  getData?(release?: boolean): Promise<unknown>;
}
export interface OrtSession {
  run(feeds: Record<string, OrtTensor>): Promise<Record<string, OrtTensor>>;
  release(): Promise<void>;
}
export interface OrtLike {
  Tensor: new (type: string, data: ArrayLike<number> | Float32Array | Int32Array, dims: readonly number[]) => OrtTensor;
}

export type Progress = (stage: string, done: number, total: number) => void;

export class CancelledError extends Error {
  constructor() {
    super('cancelled');
    this.name = 'CancelledError';
  }
}

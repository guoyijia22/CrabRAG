import { useCallback, useEffect, useRef } from "react";

interface TaskProgress { status: string }

interface TaskPollerOptions<T extends TaskProgress, R> {
  loadProgress: (runId: string) => Promise<T>;
  loadCompleted?: (runId: string, progress: T) => Promise<R>;
  onProgress: (progress: T) => void;
  onCompleted?: (result: R, progress: T) => void;
  onFailed?: (progress: T) => void;
  onError: (reason: unknown) => void;
  interval?: number;
}

interface ActiveWatcher {
  runId: string;
  token: number;
  promise: Promise<void>;
}

export function useTaskPoller<T extends TaskProgress, R = void>(options: TaskPollerOptions<T, R>) {
  const optionsRef = useRef(options);
  optionsRef.current = options;
  const mountedRef = useRef(true);
  const tokenRef = useRef(0);
  const activeRef = useRef<ActiveWatcher | null>(null);
  const waitRef = useRef<{ timer: number; resolve: () => void } | null>(null);

  const cancel = useCallback(() => {
    tokenRef.current += 1;
    activeRef.current = null;
    const waiting = waitRef.current;
    waitRef.current = null;
    if (waiting) {
      window.clearTimeout(waiting.timer);
      waiting.resolve();
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      cancel();
    };
  }, [cancel]);

  const watch = useCallback((runId: string): Promise<void> => {
    const existing = activeRef.current;
    if (existing?.runId === runId) return existing.promise;
    cancel();
    const token = tokenRef.current;
    const isCurrent = () => mountedRef.current && tokenRef.current === token;
    const promise = (async () => {
      try {
        for (;;) {
          if (!isCurrent()) return;
          const progress = await optionsRef.current.loadProgress(runId);
          if (!isCurrent()) return;
          optionsRef.current.onProgress(progress);
          if (progress.status === "completed") {
            const result = optionsRef.current.loadCompleted
              ? await optionsRef.current.loadCompleted(runId, progress)
              : undefined as R;
            if (isCurrent()) optionsRef.current.onCompleted?.(result, progress);
            return;
          }
          if (progress.status === "failed") {
            optionsRef.current.onFailed?.(progress);
            return;
          }
          await new Promise<void>((resolve) => {
            const timer = window.setTimeout(() => {
              if (waitRef.current?.timer === timer) waitRef.current = null;
              resolve();
            }, optionsRef.current.interval ?? 1000);
            waitRef.current = { timer, resolve };
          });
        }
      } catch (reason) {
        if (isCurrent()) optionsRef.current.onError(reason);
      } finally {
        if (activeRef.current?.token === token) activeRef.current = null;
      }
    })();
    activeRef.current = { runId, token, promise };
    return promise;
  }, [cancel]);

  return { watch, cancel };
}

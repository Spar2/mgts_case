import { decode, decimate } from "./numeric";
self.onmessage = (
  event: MessageEvent<{
    buffer?: ArrayBuffer;
    x?: number[];
    y?: (number | null)[];
    budget?: number;
  }>,
) => {
  try {
    if (event.data.buffer) {
      const data = decode(event.data.buffer);
      self.postMessage(data, {
        transfer: [data.instances.buffer, data.values.buffer],
      });
    } else
      self.postMessage({
        line: decimate(event.data.x!, event.data.y!, event.data.budget),
      });
  } catch (e) {
    self.postMessage({ error: String(e) });
  }
};

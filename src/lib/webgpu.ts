export async function checkWebGPU(): Promise<{ supported: boolean; reason?: string }> {
  if (!navigator.gpu) {
    return {
      supported: false,
      reason: "Din browser understøtter ikke WebGPU. Brug Chrome 113+ eller Edge 113+.",
    };
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      return {
        supported: false,
        reason: "Ingen WebGPU-adapter fundet. Prøv at opdatere din browser.",
      };
    }
    return { supported: true };
  } catch {
    return { supported: false, reason: "WebGPU kunne ikke initialiseres." };
  }
}

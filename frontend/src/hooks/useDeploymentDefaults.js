import { useEffect, useMemo, useState } from "react";
import { normalizeConfig } from "../structure/config/normalize.js";
import { aggregateModelMemory } from "../cost/aggregate.js";
import { DEFAULT_LOADS } from "../cost/defaults.js";
import { recommendSingleNodePlan } from "../cost/parallelDefaults.js";

/** 由一个 owner 统一维护架构图、Cost 和 PD 部署状态。
 * 自动推荐是派生值；只保存用户显式修改。后台 checkpoint 更新可改进默认推荐，
 * 但不能覆盖手动方案；工作负载和 what-if 控件也不会悄悄切换卡数档位。
 */
export function useDeploymentDefaults({ structure, chip, frameworkProfile }) {
  const modelKey = useMemo(() => JSON.stringify([
    structure?.source?.model_id,
    structure?.source?.revision,
    structure?.extra_config,
  ]), [structure]);
  const [override, setOverride] = useState(null);
  useEffect(() => {
    // 不把参数总量作为 key：后台延迟到达的 checkpoint 真值会改变它。
    setOverride((current) => current?.modelKey === modelKey ? current : null);
  }, [modelKey]);
  const baseline = useMemo(() => {
    const config = normalizeConfig(structure?.extra_config || {});
    const { memory } = aggregateModelMemory({
      graph: structure?.graph, config,
      parameterCount: structure?.summary?.parameters_by_dtype,
      ...DEFAULT_LOADS.prefill, frameworkProfile,
    });
    return { config, accounting: memory.accounting };
  }, [structure, frameworkProfile]);
  const recommendation = useMemo(() => recommendSingleNodePlan({
    graph: structure?.graph, ...baseline, chip,
  }), [structure, baseline, chip]);
  const manual = override?.modelKey === modelKey && override.deployment != null;
  const deployment = manual ? override.deployment : recommendation;
  const change = (patch) => setOverride((current) => {
    const base = current?.modelKey === modelKey ? current.deployment : deployment;
    // NumberInput 即使没有编辑也会在 blur 时 commit；仅聚焦字段不能关闭自动推荐，
    // 否则下一次切换硬件时会错误地保留旧方案。
    if (Object.entries(patch).every(([key, value]) => JSON.stringify(value) === JSON.stringify(base[key]))) return current;
    return {
      modelKey,
      deployment: { plans: base.plans, nodes: base.nodes, gpusPerNode: base.gpusPerNode, ...patch },
    };
  });
  return {
    ...deployment,
    recommendation, manual,
    setPlans: (plans) => change({ plans }),
    setNodes: (nodes) => change({ nodes }),
    setGpusPerNode: (gpusPerNode) => change({ gpusPerNode }),
    reset: () => setOverride(null),
  };
}

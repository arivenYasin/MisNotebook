import { useState, useEffect, useCallback, useRef, useMemo } from "react";
import * as d3 from "d3";

// ============ CONSTANTS & CONFIG ============
const CONFIG = {
  lambda: 0.95, eta: 0.4, k: 2.5, w: 4,
  explorePct: 0.2, batchSize: 10, minInterval: 3,
  sigmoidAlpha: 0.7,
  leitnerIntervals: [1, 2, 4, 8, 16, 32],
};

// ============ ALGORITHM HELPERS ============
function mastery(errorCount) {
  if (errorCount <= 0) return 1.0;
  return 2 / (1 + Math.exp(errorCount * CONFIG.sigmoidAlpha));
}

function masteryColor(m) {
  if (m > 0.7) return `hsl(${120 * (m - 0.7) / 0.3 + 40}, 75%, 45%)`;
  if (m > 0.3) return `hsl(${40 * (m - 0.3) / 0.4}, 80%, 50%)`;
  return `hsl(0, ${70 + 30 * (0.3 - m) / 0.3}%, ${45 + 10 * (0.3 - m) / 0.3}%)`;
}

function updateConfidence(C, M_before, isCorrect, F, M_current, stepsSinceLast) {
  const dt = Math.max(stepsSinceLast, 0);
  const decayExp = 1 + CONFIG.k * F * M_current;
  let newC = C * Math.pow(Math.pow(CONFIG.lambda, decayExp), dt);
  const consistency = isCorrect ? M_before : (1 - M_before);
  newC = (1 - CONFIG.eta) * newC + CONFIG.eta * consistency;
  return Math.max(0, Math.min(1, newC));
}

function priority(M, C) {
  return M * (1 - C) + CONFIG.w * M * (1 - M);
}

function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).substr(2, 6);
}

// ============ STORAGE ============
async function saveData(nodes, questions, state) {
  try {
    localStorage.setItem('kg-data', JSON.stringify({ nodes, questions, state }));
  } catch (e) { console.error('Save failed:', e); }
}

async function loadData() {
  try {
    const raw = localStorage.getItem('kg-data');
    return raw ? JSON.parse(raw) : null;
  } catch (e) { return null; }
}

// ============ INITIAL STATE ============
const INIT_STATE = { globalStep: 0, recentQuestions: [] };

// ============ SAMPLE DATA GENERATOR ============
function generateSampleData() {
  const nid = (i) => `n${i}`;
  const qid = (i) => `q${i}`;
  let n = 0, q = 0;

  const mkNode = (name, parentId, overrides) => {
    const id = nid(++n);
    return { id, name, parentId, confidence: 0, totalChecks: 0, totalErrors: 0, lastCheckStep: 0, forgetTendency: 0, ...overrides };
  };
  const mkQ = (content, kpIds, diff, status, failedKPs, step) => {
    const id = qid(++q);
    return { id, content, knowledgePointIds: kpIds, difficulty: diff, status, failedKnowledgePointIds: failedKPs || [], history: [{ step: step || 0, result: status === 'wrong' ? 'wrong' : 'correct', failedKPs: failedKPs || [] }], leitnerBox: status === 'wrong' ? 0 : 2, lastReviewStep: step || 0 };
  };

  const nodes = {};
  const questions = {};
  const add = (nd) => { nodes[nd.id] = nd; return nd.id; };
  const addQ = (qq) => { questions[qq.id] = qq; return qq.id; };

  // Chapter 1: 极限
  const c1 = add(mkNode('极限', null));
  const c1_1 = add(mkNode('函数极限', c1));
  const c1_1_1 = add(mkNode('极限定义', c1_1));
  const c1_1_2 = add(mkNode('极限运算法则', c1_1));
  const c1_1_3 = add(mkNode('洛必达法则', c1_1));
  const c1_1_3a = add(mkNode('洛必达使用条件', c1_1_3));
  const c1_1_3b = add(mkNode('洛必达与泰勒配合', c1_1_3));
  const c1_2 = add(mkNode('数列极限', c1));
  const c1_2_1 = add(mkNode('单调有界准则', c1_2));
  const c1_2_2 = add(mkNode('夹逼准则', c1_2));
  const c1_3 = add(mkNode('无穷小与无穷大', c1));
  const c1_3_1 = add(mkNode('等价无穷小替换', c1_3));
  const c1_3_2 = add(mkNode('无穷小比较', c1_3));

  // Chapter 2: 一元函数微分学
  const c2 = add(mkNode('一元函数微分学', null));
  const c2_1 = add(mkNode('导数定义', c2));
  const c2_1_1 = add(mkNode('导数的几何意义', c2_1));
  const c2_1_2 = add(mkNode('可导与连续的关系', c2_1));
  const c2_2 = add(mkNode('求导法则', c2));
  const c2_2_1 = add(mkNode('复合函数求导', c2_2));
  const c2_2_2 = add(mkNode('隐函数求导', c2_2));
  const c2_2_3 = add(mkNode('参数方程求导', c2_2));
  const c2_3 = add(mkNode('中值定理', c2));
  const c2_3_1 = add(mkNode('罗尔定理', c2_3));
  const c2_3_2 = add(mkNode('拉格朗日中值定理', c2_3));
  const c2_3_3 = add(mkNode('柯西中值定理', c2_3));

  // Chapter 3: 一元函数积分学
  const c3 = add(mkNode('一元函数积分学', null));
  const c3_1 = add(mkNode('不定积分', c3));
  const c3_1_1 = add(mkNode('换元积分法', c3_1));
  const c3_1_2 = add(mkNode('分部积分法', c3_1));
  const c3_2 = add(mkNode('定积分', c3));
  const c3_2_1 = add(mkNode('牛顿-莱布尼茨公式', c3_2));
  const c3_2_2 = add(mkNode('变限积分求导', c3_2));
  const c3_2_3 = add(mkNode('定积分的应用', c3_2));

  // Chapter 4: 多元函数微分学
  const c4 = add(mkNode('多元函数微分学', null));
  const c4_1 = add(mkNode('偏导数', c4));
  const c4_1_1 = add(mkNode('偏导数定义', c4_1));
  const c4_1_2 = add(mkNode('全微分', c4_1));
  const c4_2 = add(mkNode('多元复合函数求导', c4));
  const c4_3 = add(mkNode('极值与最值', c4));
  const c4_3_1 = add(mkNode('无条件极值', c4_3));
  const c4_3_2 = add(mkNode('拉格朗日乘数法', c4_3));

  // Chapter 5: 线性代数
  const c5 = add(mkNode('线性代数', null));
  const c5_1 = add(mkNode('行列式', c5));
  const c5_1_1 = add(mkNode('行列式性质', c5_1));
  const c5_1_2 = add(mkNode('行列式展开', c5_1));
  const c5_2 = add(mkNode('矩阵', c5));
  const c5_2_1 = add(mkNode('矩阵运算', c5_2));
  const c5_2_2 = add(mkNode('逆矩阵', c5_2));
  const c5_2_3 = add(mkNode('矩阵的秩', c5_2));
  const c5_3 = add(mkNode('线性方程组', c5));
  const c5_3_1 = add(mkNode('齐次方程组', c5_3));
  const c5_3_2 = add(mkNode('非齐次方程组', c5_3));
  const c5_4 = add(mkNode('特征值与特征向量', c5));
  const c5_4_1 = add(mkNode('特征值计算', c5_4));
  const c5_4_2 = add(mkNode('相似对角化', c5_4));

  // ---- Questions ----
  // 极限 - mostly mastered
  addQ(mkQ('660 T1-01: 求 lim(x→0) sin(x)/x', [c1_1_1], 1, 'correct', [], 5));
  addQ(mkQ('660 T1-02: ε-δ语言证明极限', [c1_1_1], 4, 'correct', [], 8));
  addQ(mkQ('660 T1-03: 四则运算求极限', [c1_1_2], 2, 'correct', [], 10));
  addQ(mkQ('660 T1-04: 洛必达求 0/0 型极限', [c1_1_3a], 2, 'correct', [], 12));
  addQ(mkQ('660 T1-05: 洛必达使用条件辨析（陷阱题）', [c1_1_3a], 4, 'wrong', [c1_1_3a], 15));
  addQ(mkQ('660 T1-06: 泰勒展开配合洛必达', [c1_1_3b], 3, 'wrong', [c1_1_3b], 14));
  addQ(mkQ('660 T1-07: 泰勒+洛必达综合题', [c1_1_3b, c1_1_3a], 5, 'wrong', [c1_1_3b], 18));
  addQ(mkQ('1000题 T1-08: 数列极限-单调有界', [c1_2_1], 3, 'correct', [], 20));
  addQ(mkQ('1000题 T1-09: 夹逼准则应用', [c1_2_2], 3, 'correct', [], 22));
  addQ(mkQ('1000题 T1-10: 等价无穷小替换', [c1_3_1], 2, 'correct', [], 25));
  addQ(mkQ('真题2018: 等价无穷小综合', [c1_3_1, c1_1_2], 4, 'correct', [], 28));
  addQ(mkQ('1000题 T1-12: 无穷小阶数比较', [c1_3_2], 3, 'wrong', [c1_3_2], 30));

  // 一元微分 - mixed
  addQ(mkQ('660 T2-01: 导数定义求导', [c2_1_1], 2, 'correct', [], 32));
  addQ(mkQ('660 T2-02: 导数的几何意义-切线', [c2_1_1], 2, 'correct', [], 33));
  addQ(mkQ('660 T2-03: 可导与连续辨析', [c2_1_2], 3, 'wrong', [c2_1_2], 35));
  addQ(mkQ('660 T2-04: 复合函数求导链式法则', [c2_2_1], 2, 'correct', [], 36));
  addQ(mkQ('1000题 T2-05: 三重复合求导', [c2_2_1], 4, 'wrong', [c2_2_1], 38));
  addQ(mkQ('660 T2-06: 隐函数求导', [c2_2_2], 3, 'correct', [], 40));
  addQ(mkQ('真题2020: 隐函数二阶导', [c2_2_2], 4, 'wrong', [c2_2_2], 42));
  addQ(mkQ('660 T2-08: 参数方程求导', [c2_2_3], 3, 'correct', [], 44));
  addQ(mkQ('660 T2-09: 罗尔定理证明题', [c2_3_1], 4, 'wrong', [c2_3_1], 46));
  addQ(mkQ('1000题 T2-10: 罗尔定理构造辅助函数', [c2_3_1], 5, 'wrong', [c2_3_1], 48));
  addQ(mkQ('660 T2-11: 拉格朗日中值定理应用', [c2_3_2], 3, 'correct', [], 50));
  addQ(mkQ('真题2019: 拉格朗日+罗尔综合证明', [c2_3_2, c2_3_1], 5, 'wrong', [c2_3_1, c2_3_2], 52));
  addQ(mkQ('1000题 T2-13: 柯西中值定理', [c2_3_3], 4, 'wrong', [c2_3_3], 54));

  // 积分 - many errors
  addQ(mkQ('660 T3-01: 第一类换元', [c3_1_1], 2, 'correct', [], 55));
  addQ(mkQ('660 T3-02: 第二类换元-三角', [c3_1_1], 3, 'wrong', [c3_1_1], 56));
  addQ(mkQ('1000题 T3-03: 复杂换元积分', [c3_1_1], 4, 'wrong', [c3_1_1], 58));
  addQ(mkQ('660 T3-04: 分部积分基础', [c3_1_2], 2, 'correct', [], 60));
  addQ(mkQ('1000题 T3-05: 分部积分-表格法', [c3_1_2], 3, 'wrong', [c3_1_2], 62));
  addQ(mkQ('真题2021: 分部积分循环积分', [c3_1_2], 4, 'wrong', [c3_1_2], 64));
  addQ(mkQ('660 T3-07: 牛莱公式直接计算', [c3_2_1], 2, 'correct', [], 65));
  addQ(mkQ('660 T3-08: 变限积分求导', [c3_2_2], 3, 'wrong', [c3_2_2], 66));
  addQ(mkQ('1000题 T3-09: 变限积分+链式法则', [c3_2_2, c2_2_1], 4, 'wrong', [c3_2_2], 68));
  addQ(mkQ('真题2022: 定积分求面积', [c3_2_3], 3, 'correct', [], 70));
  addQ(mkQ('真题2017: 定积分求旋转体体积', [c3_2_3], 4, 'wrong', [c3_2_3], 72));

  // 多元微分 - weak area
  addQ(mkQ('660 T4-01: 偏导数定义计算', [c4_1_1], 2, 'correct', [], 74));
  addQ(mkQ('660 T4-02: 偏导数存在性与连续性', [c4_1_1], 4, 'wrong', [c4_1_1], 75));
  addQ(mkQ('1000题 T4-03: 全微分判断', [c4_1_2], 3, 'wrong', [c4_1_2], 76));
  addQ(mkQ('1000题 T4-04: 全微分与偏导综合', [c4_1_2, c4_1_1], 4, 'wrong', [c4_1_2, c4_1_1], 78));
  addQ(mkQ('660 T4-05: 多元复合求导-链式法则', [c4_2], 3, 'wrong', [c4_2], 80));
  addQ(mkQ('真题2020: 多元复合函数偏导', [c4_2], 4, 'wrong', [c4_2], 82));
  addQ(mkQ('660 T4-07: 无条件极值判定', [c4_3_1], 3, 'wrong', [c4_3_1], 84));
  addQ(mkQ('1000题 T4-08: 拉格朗日乘数法', [c4_3_2], 4, 'wrong', [c4_3_2], 86));
  addQ(mkQ('真题2019: 拉格朗日乘数法应用', [c4_3_2], 5, 'wrong', [c4_3_2], 88));

  // 线代 - partially mastered
  addQ(mkQ('660 T5-01: 行列式性质', [c5_1_1], 2, 'correct', [], 90));
  addQ(mkQ('660 T5-02: 按行展开计算行列式', [c5_1_2], 2, 'correct', [], 91));
  addQ(mkQ('1000题 T5-03: 高阶行列式', [c5_1_2], 4, 'wrong', [c5_1_2], 92));
  addQ(mkQ('660 T5-04: 矩阵乘法', [c5_2_1], 2, 'correct', [], 93));
  addQ(mkQ('660 T5-05: 逆矩阵求解', [c5_2_2], 3, 'correct', [], 94));
  addQ(mkQ('1000题 T5-06: 逆矩阵性质证明', [c5_2_2], 4, 'wrong', [c5_2_2], 95));
  addQ(mkQ('660 T5-07: 矩阵秩的判断', [c5_2_3], 3, 'wrong', [c5_2_3], 96));
  addQ(mkQ('660 T5-08: 齐次方程组基础解系', [c5_3_1], 3, 'correct', [], 97));
  addQ(mkQ('1000题 T5-09: 齐次方程组解空间维数', [c5_3_1], 4, 'wrong', [c5_3_1], 98));
  addQ(mkQ('660 T5-10: 非齐次方程组通解', [c5_3_2], 3, 'wrong', [c5_3_2], 99));
  addQ(mkQ('真题2021: 非齐次方程组解的结构', [c5_3_2, c5_3_1], 4, 'wrong', [c5_3_2], 100));
  addQ(mkQ('660 T5-12: 特征值计算', [c5_4_1], 3, 'correct', [], 101));
  addQ(mkQ('1000题 T5-13: 特征值性质', [c5_4_1], 3, 'wrong', [c5_4_1], 102));
  addQ(mkQ('660 T5-14: 相似对角化条件', [c5_4_2], 3, 'wrong', [c5_4_2], 103));
  addQ(mkQ('真题2022: 相似对角化综合', [c5_4_2, c5_4_1], 5, 'wrong', [c5_4_2, c5_4_1], 105));

  // Update node stats based on questions
  Object.values(questions).forEach(qq => {
    const affectedKPs = qq.status === 'wrong' ? (qq.failedKnowledgePointIds.length > 0 ? qq.failedKnowledgePointIds : qq.knowledgePointIds) : qq.knowledgePointIds;
    affectedKPs.forEach(kpId => {
      if (nodes[kpId]) {
        nodes[kpId].totalChecks += 1;
        if (qq.status === 'wrong') nodes[kpId].totalErrors += 1;
        nodes[kpId].forgetTendency = nodes[kpId].totalErrors / Math.max(nodes[kpId].totalChecks, 1);
        const errCount = Object.values(questions).filter(x => x.status === 'wrong' && x.knowledgePointIds.includes(kpId)).length;
        const M_before = mastery(errCount);
        nodes[kpId].confidence = updateConfidence(nodes[kpId].confidence, M_before, qq.status === 'correct', nodes[kpId].forgetTendency, M_before, 5);
        nodes[kpId].lastCheckStep = qq.history[0]?.step || 0;
      }
    });
  });

  return { nodes, questions, state: { globalStep: 110, recentQuestions: [] } };
}

// ============ MAIN APP ============
export default function App() {
  const [nodes, setNodes] = useState({});
  const [questions, setQuestions] = useState({});
  const [appState, setAppState] = useState(INIT_STATE);
  const [selectedNodeId, setSelectedNodeId] = useState(null);
  const [view, setView] = useState('graph'); // graph | review | addQuestion
  const [reviewQueue, setReviewQueue] = useState([]);
  const [reviewIdx, setReviewIdx] = useState(0);
  const [loaded, setLoaded] = useState(false);
  const [sidePanel, setSidePanel] = useState('tree'); // tree | nodeDetail | addNode | addQuestion
  const [editingNode, setEditingNode] = useState(null);
  const [newNodeName, setNewNodeName] = useState('');
  const [newQ, setNewQ] = useState({ content: '', knowledgePointIds: [], difficulty: 3 });
  const [expandedNodes, setExpandedNodes] = useState(new Set());
  const [reviewResult, setReviewResult] = useState(null);
  const [failedKPs, setFailedKPs] = useState([]);
  const [graphKPSelectMode, setGraphKPSelectMode] = useState(false);
  const [graphKPIds, setGraphKPIds] = useState([]);

  // Load data
  useEffect(() => {
    loadData().then(d => {
      if (d) {
        setNodes(d.nodes || {});
        setQuestions(d.questions || {});
        setAppState(d.state || INIT_STATE);
      }
      setLoaded(true);
    });
  }, []);

  // Save on change
  useEffect(() => {
    if (loaded) saveData(nodes, questions, appState);
  }, [nodes, questions, appState, loaded]);

  // ---- Node helpers ----
  const rootNodes = useMemo(() => 
    Object.values(nodes).filter(n => !n.parentId).sort((a,b) => a.name.localeCompare(b.name)),
    [nodes]
  );

  const getChildren = useCallback((id) => 
    Object.values(nodes).filter(n => n.parentId === id).sort((a,b) => a.name.localeCompare(b.name)),
    [nodes]
  );

  const isLeaf = useCallback((id) => getChildren(id).length === 0, [getChildren]);

  const getAllLeafIds = useCallback((id) => {
    if (isLeaf(id)) return [id];
    return getChildren(id).flatMap(c => getAllLeafIds(c.id));
  }, [isLeaf, getChildren]);

  const getSubtreeErrorCount = useCallback((id) => {
    const leafIds = getAllLeafIds(id);
    let count = 0;
    Object.values(questions).forEach(q => {
      if (q.status === 'wrong') {
        if (q.knowledgePointIds.some(kp => leafIds.includes(kp))) count++;
      }
    });
    return count;
  }, [getAllLeafIds, questions]);

  const getNodeErrorCount = useCallback((id) => {
    if (!isLeaf(id)) return getSubtreeErrorCount(id);
    return Object.values(questions).filter(q => 
      q.status === 'wrong' && q.knowledgePointIds.includes(id)
    ).length;
  }, [isLeaf, getSubtreeErrorCount, questions]);

  const getNodeMastery = useCallback((id) => mastery(getNodeErrorCount(id)), [getNodeErrorCount]);

  const getNodeConfidence = useCallback((id) => {
    return nodes[id]?.confidence ?? 0;
  }, [nodes]);

  const getNodeQuestions = useCallback((id) => {
    // For leaf nodes: only questions directly associated
    if (isLeaf(id)) {
      return Object.values(questions).filter(q => q.knowledgePointIds.includes(id));
    }
    // For non-leaf: include questions from all descendants (including "XX本身" children)
    const allIds = new Set([id, ...getAllLeafIds(id)]);
    return Object.values(questions).filter(q => q.knowledgePointIds.some(k => allIds.has(k)));
  }, [questions, isLeaf, getAllLeafIds]);

  const allLeafNodes = useMemo(() => 
    Object.values(nodes).filter(n => isLeaf(n.id)),
    [nodes, isLeaf]
  );

  // ---- Add node ----
  const addNode = useCallback((parentId, name) => {
    const id = generateId();
    setNodes(prev => ({
      ...prev,
      [id]: {
        id, name, parentId: parentId || null,
        confidence: 0, totalChecks: 0, totalErrors: 0,
        lastCheckStep: appState.globalStep, forgetTendency: 0,
      }
    }));
    setExpandedNodes(prev => {
      const n = new Set(prev);
      if (parentId) n.add(parentId);
      return n;
    });
    return id;
  }, [appState.globalStep]);

  const deleteNode = useCallback((id) => {
    const toDelete = new Set();
    const collect = (nid) => {
      toDelete.add(nid);
      getChildren(nid).forEach(c => collect(c.id));
    };
    collect(id);
    setNodes(prev => {
      const next = { ...prev };
      toDelete.forEach(d => delete next[d]);
      return next;
    });
    setQuestions(prev => {
      const next = { ...prev };
      Object.keys(next).forEach(qid => {
        next[qid] = { ...next[qid], knowledgePointIds: next[qid].knowledgePointIds.filter(k => !toDelete.has(k)) };
        if (next[qid].knowledgePointIds.length === 0) delete next[qid];
      });
      return next;
    });
    if (toDelete.has(selectedNodeId)) setSelectedNodeId(null);
  }, [getChildren, selectedNodeId]);

  const deleteQuestion = useCallback((qId) => {
    setQuestions(prev => {
      const next = { ...prev };
      delete next[qId];
      return next;
    });
  }, []);

  // ---- Add question ----
  const addQuestion = useCallback((content, kpIds, difficulty, status, failedKPIds) => {
    const id = generateId();
    const step = appState.globalStep;
    const q = {
      id, content, knowledgePointIds: kpIds, difficulty,
      status: status || 'untested',
      failedKnowledgePointIds: failedKPIds || [],
      history: [],
      leitnerBox: 0, lastReviewStep: step,
    };
    if (status === 'wrong') {
      q.history.push({ step, result: 'wrong', failedKPs: failedKPIds || kpIds });
    } else if (status === 'correct') {
      q.history.push({ step, result: 'correct', failedKPs: [] });
    }

    // Use functional update to avoid clobbering concurrent addNode updates
    const affectedKPs = status === 'wrong' ? (failedKPIds || kpIds) : (status === 'correct' ? kpIds : []);
    if (affectedKPs.length > 0) {
      setNodes(prev => {
        const next = { ...prev };
        affectedKPs.forEach(kpId => {
          if (!next[kpId]) return;
          next[kpId] = { ...next[kpId] };
          next[kpId].totalChecks = (next[kpId].totalChecks || 0) + 1;
          if (status === 'wrong') {
            next[kpId].totalErrors = (next[kpId].totalErrors || 0) + 1;
          }
          const F = (next[kpId].totalErrors || 0) / Math.max(next[kpId].totalChecks, 1);
          next[kpId].forgetTendency = F;
          // Compute error count from current questions + the new one
          const existingErrors = Object.values(questions).filter(
            qq => qq.status === 'wrong' && qq.knowledgePointIds.includes(kpId)
          ).length;
          const newError = (status === 'wrong' && kpIds.includes(kpId)) ? 1 : 0;
          const M_before = mastery(existingErrors);
          const M_after = mastery(existingErrors + newError);
          next[kpId].confidence = updateConfidence(
            next[kpId].confidence, M_before, status === 'correct', F, M_after,
            step - (next[kpId].lastCheckStep || 0)
          );
          next[kpId].lastCheckStep = step;
        });
        return next;
      });
    }
    setQuestions(prev => ({ ...prev, [id]: q }));
    setAppState(prev => ({ ...prev, globalStep: prev.globalStep + 1 }));
    return id;
  }, [appState.globalStep, questions]);

  // ---- Review: answer a question ----
  const answerQuestion = useCallback((qId, isCorrect, failedKPIds) => {
    const q = questions[qId];
    if (!q) return;
    const newQuestions = { ...questions };
    const newNodes = { ...nodes };
    const wasWrong = q.status === 'wrong';
    
    newQuestions[qId] = { ...q };
    newQuestions[qId].history = [...q.history, {
      step: appState.globalStep, result: isCorrect ? 'correct' : 'wrong',
      failedKPs: failedKPIds || []
    }];

    if (isCorrect) {
      newQuestions[qId].status = 'correct';
      newQuestions[qId].failedKnowledgePointIds = [];
      newQuestions[qId].leitnerBox = Math.min((q.leitnerBox || 0) + 1, CONFIG.leitnerIntervals.length - 1);
    } else {
      newQuestions[qId].status = 'wrong';
      newQuestions[qId].failedKnowledgePointIds = failedKPIds || q.knowledgePointIds;
      newQuestions[qId].leitnerBox = 0;
    }
    newQuestions[qId].lastReviewStep = appState.globalStep;

    // Update knowledge point stats
    const affectedKPs = isCorrect ? q.knowledgePointIds : (failedKPIds || q.knowledgePointIds);
    affectedKPs.forEach(kpId => {
      if (newNodes[kpId]) {
        newNodes[kpId] = { ...newNodes[kpId] };
        const errorCountBefore = Object.values(questions).filter(
          qq => qq.status === 'wrong' && qq.knowledgePointIds.includes(kpId)
        ).length;
        const M_before = mastery(errorCountBefore);
        
        newNodes[kpId].totalChecks = (newNodes[kpId].totalChecks || 0) + 1;
        if (!isCorrect) newNodes[kpId].totalErrors = (newNodes[kpId].totalErrors || 0) + 1;
        const F = (newNodes[kpId].totalErrors || 0) / Math.max(newNodes[kpId].totalChecks, 1);
        newNodes[kpId].forgetTendency = F;
        newNodes[kpId].confidence = updateConfidence(
          newNodes[kpId].confidence, M_before, isCorrect, F,
          mastery(isCorrect && wasWrong ? errorCountBefore - 1 : (!isCorrect && !wasWrong ? errorCountBefore + 1 : errorCountBefore)),
          appState.globalStep - (newNodes[kpId].lastCheckStep || 0)
        );
        newNodes[kpId].lastCheckStep = appState.globalStep;
      }
    });

    setQuestions(newQuestions);
    setNodes(newNodes);
    setAppState(prev => ({
      ...prev, globalStep: prev.globalStep + 1,
      recentQuestions: [...(prev.recentQuestions || []).slice(-50), qId]
    }));
  }, [questions, nodes, appState]);

  // ---- Generate review queue ----
  const generateReview = useCallback(() => {
    const step = appState.globalStep;
    const recent = new Set(appState.recentQuestions?.slice(-CONFIG.minInterval) || []);
    const allQs = Object.values(questions);
    if (allQs.length === 0) return;

    // Source 1: Leitner due questions
    const dueQs = allQs.filter(q => {
      if (recent.has(q.id)) return false;
      if (q.status !== 'wrong') return false;
      const interval = CONFIG.leitnerIntervals[q.leitnerBox || 0] || 1;
      return (step - (q.lastReviewStep || 0)) >= interval;
    });

    // Source 2: Priority-driven
    const leafPriorities = allLeafNodes.map(n => ({
      id: n.id, p: priority(getNodeMastery(n.id), getNodeConfidence(n.id))
    })).sort((a, b) => b.p - a.p);

    // Source 3: Exploration (lowest confidence)
    const leafByConfidence = allLeafNodes
      .filter(n => getNodeQuestions(n.id).length > 0)
      .sort((a, b) => (a.confidence || 0) - (b.confidence || 0));

    const queue = [];
    const usedQIds = new Set();

    // Add due questions first
    dueQs.slice(0, CONFIG.batchSize).forEach(q => {
      if (!usedQIds.has(q.id)) { queue.push(q.id); usedQIds.add(q.id); }
    });

    const remaining = CONFIG.batchSize - queue.length;
    const exploreCount = Math.ceil(remaining * CONFIG.explorePct);
    const priorityCount = remaining - exploreCount;

    // Priority questions
    let added = 0;
    for (const lp of leafPriorities) {
      if (added >= priorityCount) break;
      const qs = getNodeQuestions(lp.id).filter(q => !usedQIds.has(q.id) && !recent.has(q.id));
      if (qs.length > 0) {
        const pick = qs[Math.floor(Math.random() * qs.length)];
        queue.push(pick.id); usedQIds.add(pick.id); added++;
      }
    }

    // Explore questions
    let exploreAdded = 0;
    for (const ln of leafByConfidence) {
      if (exploreAdded >= exploreCount) break;
      const qs = getNodeQuestions(ln.id).filter(q => !usedQIds.has(q.id) && !recent.has(q.id));
      if (qs.length > 0) {
        const pick = qs[Math.floor(Math.random() * qs.length)];
        queue.push(pick.id); usedQIds.add(pick.id); exploreAdded++;
      }
    }

    // Fill remaining if needed
    if (queue.length < CONFIG.batchSize) {
      const remaining2 = allQs.filter(q => !usedQIds.has(q.id) && !recent.has(q.id));
      while (queue.length < CONFIG.batchSize && remaining2.length > 0) {
        const idx = Math.floor(Math.random() * remaining2.length);
        queue.push(remaining2[idx].id);
        usedQIds.add(remaining2[idx].id);
        remaining2.splice(idx, 1);
      }
    }

    setReviewQueue(queue);
    setReviewIdx(0);
    setView('review');
    setReviewResult(null);
    setFailedKPs([]);
  }, [appState, questions, allLeafNodes, getNodeMastery, getNodeConfidence, getNodeQuestions]);

  // ---- Graph data for d3 ----
  // Detect "XX本身" abstract nodes - they should be invisible
  const isSelfNode = useCallback((id) => {
    const n = nodes[id];
    if (!n || !n.parentId) return false;
    const parent = nodes[n.parentId];
    return parent && n.name === parent.name + '本身';
  }, [nodes]);

  const graphData = useMemo(() => {
    const visibleNodes = [];
    const links = [];
    const visited = new Set();
    const visibleAncestor = {};

    const addNode = (id, depth) => {
      if (visited.has(id) || !nodes[id]) return;
      visited.add(id);
      const n = nodes[id];
      const children = getChildren(id);
      // Filter out "XX本身" from visible children
      const visibleChildren = children.filter(c => !isSelfNode(c.id));
      const selfChildren = children.filter(c => isSelfNode(c.id));
      const leaf = isLeaf(id);
      const isExpanded = expandedNodes.has(id);
      // A node is effectively a leaf if its only children are "XX本身" nodes
      const effectiveLeaf = leaf || (children.length > 0 && visibleChildren.length === 0);
      const showSubtree = !effectiveLeaf && !isExpanded;
      
      // Map self to self
      visibleAncestor[id] = id;
      // Always map "XX本身" children to parent
      selfChildren.forEach(c => {
        visibleAncestor[c.id] = id;
        visited.add(c.id);
      });
      // Map hidden descendants when collapsed
      if (showSubtree) {
        const mapDescendants = (nid) => {
          getChildren(nid).forEach(c => {
            visibleAncestor[c.id] = id;
            if (!visited.has(c.id)) {
              visited.add(c.id);
              mapDescendants(c.id);
            }
          });
        };
        mapDescendants(id);
      }

      // Error count: include self + "XX本身" children always, + full subtree when collapsed
      let errorCount;
      // IDs whose questions should be counted as belonging to this node
      const selfIds = [id, ...selfChildren.map(c => c.id)];
      if (showSubtree) {
        errorCount = getSubtreeErrorCount(id);
      } else {
        errorCount = Object.values(questions).filter(q => 
          q.status === 'wrong' && q.knowledgePointIds.some(k => selfIds.includes(k))
        ).length;
      }
      const m = mastery(errorCount);
      
      let qCount;
      if (showSubtree) {
        qCount = getAllLeafIds(id).reduce((s, lid) => s + getNodeQuestions(lid).length, 0);
      } else {
        // Count questions linked to self or "XX本身" children
        qCount = Object.values(questions).filter(q =>
          q.knowledgePointIds.some(k => selfIds.includes(k))
        ).length;
      }

      // Always use full subtree question count for radius so parent nodes stay larger than children
      const totalSubtreeQCount = getNodeQuestions(id).length;
      visibleNodes.push({
        id, name: n.name, mastery: m, depth,
        isLeaf: effectiveLeaf, confidence: getNodeConfidence(id),
        errorCount, qCount, isExpanded: effectiveLeaf ? false : isExpanded, showSubtree,
        radius: Math.max(14, Math.min(50, 14 + Math.sqrt(totalSubtreeQCount) * 4.5)),
      });

      if (isExpanded && !effectiveLeaf) {
        visibleChildren.forEach(c => {
          addNode(c.id, depth + 1);
          links.push({ source: id, target: c.id, type: 'parent' });
        });
      }
    };

    rootNodes.forEach(r => addNode(r.id, 0));

    // Cross-links: resolve each question's KPs to their visible ancestors
    const kpPairs = {};
    Object.values(questions).forEach(q => {
      // Map each kp to its visible ancestor
      const visKps = [...new Set(
        q.knowledgePointIds.map(k => visibleAncestor[k]).filter(Boolean)
      )];
      for (let i = 0; i < visKps.length; i++) {
        for (let j = i + 1; j < visKps.length; j++) {
          if (visKps[i] === visKps[j]) continue; // same visible node
          const key = [visKps[i], visKps[j]].sort().join('-');
          kpPairs[key] = (kpPairs[key] || 0) + 1;
        }
      }
    });
    Object.entries(kpPairs).forEach(([key, count]) => {
      const [s, t] = key.split('-');
      // Don't add if already connected by parent link
      const hasParentLink = links.some(l => 
        (l.source === s && l.target === t) || (l.source === t && l.target === s));
      if (!hasParentLink) {
        links.push({ source: s, target: t, type: 'question', count });
      }
    });

    return { nodes: visibleNodes, links, visibleAncestor };
  }, [nodes, questions, expandedNodes, rootNodes, getChildren, isLeaf, isSelfNode, getNodeMastery, 
      getNodeConfidence, getNodeErrorCount, getSubtreeErrorCount, getNodeQuestions, getAllLeafIds]);

  // ---- Current review question ----
  const currentReviewQ = reviewQueue[reviewIdx] ? questions[reviewQueue[reviewIdx]] : null;

  if (!loaded) return <div style={styles.loading}>加载中...</div>;

  return (
    <div style={styles.container}>
      {/* Top Bar */}
      <div style={styles.topBar}>
        <div style={styles.logo}>
          <span style={styles.logoIcon}>◈</span> 知识图谱复习系统
        </div>
        <div style={styles.topActions}>
          <span style={styles.statBadge}>
            知识点 {Object.keys(nodes).length} · 题目 {Object.keys(questions).length}
          </span>
          <button style={styles.sampleBtn} onClick={() => {
            const d = generateSampleData();
            setNodes(d.nodes); setQuestions(d.questions); setAppState(d.state);
            setExpandedNodes(new Set(Object.values(d.nodes).filter(nn => !nn.parentId).map(nn => nn.id)));
            setSelectedNodeId(null);
          }}>
            ◆ 加载示例数据
          </button>
          <button style={styles.resetBtn} onClick={() => {
            setNodes({}); setQuestions({}); setAppState(INIT_STATE);
            setSelectedNodeId(null); setExpandedNodes(new Set());
          }}>
            清空数据
          </button>
          <button style={styles.reviewBtn} onClick={generateReview}
            disabled={Object.keys(questions).length === 0}>
            ▶ 开始复习
          </button>
        </div>
      </div>

      <div style={styles.mainLayout}>
        {/* Side Panel */}
        <div style={styles.sidePanel}>
          <div style={styles.sideTabs}>
            <button style={sidePanel === 'tree' ? styles.sideTabActive : styles.sideTab}
              onClick={() => setSidePanel('tree')}>知识树</button>
            <button style={sidePanel === 'addNode' ? styles.sideTabActive : styles.sideTab}
              onClick={() => setSidePanel('addNode')}>+ 知识点</button>
            <button style={sidePanel === 'addQuestion' ? styles.sideTabActive : styles.sideTab}
              onClick={() => setSidePanel('addQuestion')}>+ 题目</button>
          </div>

          <div style={styles.sidePanelContent}>
            {sidePanel === 'tree' && (
              <TreeView nodes={nodes} rootNodes={rootNodes} getChildren={getChildren}
                isLeaf={isLeaf} isSelfNode={isSelfNode} expandedNodes={expandedNodes} setExpandedNodes={setExpandedNodes}
                selectedNodeId={selectedNodeId} setSelectedNodeId={(id) => { setSelectedNodeId(id); setSidePanel('tree'); }}
                getNodeMastery={getNodeMastery} getNodeConfidence={getNodeConfidence}
                getNodeErrorCount={getNodeErrorCount} getNodeQuestions={getNodeQuestions}
                deleteNode={deleteNode} />
            )}
            {sidePanel === 'addNode' && (
              <AddNodePanel nodes={nodes} rootNodes={rootNodes} getChildren={getChildren}
                expandedNodes={expandedNodes} setExpandedNodes={setExpandedNodes}
                addNode={addNode} onDone={() => setSidePanel('tree')} selectedNodeId={selectedNodeId} />
            )}
            {sidePanel === 'addQuestion' && (
              <AddQuestionPanel nodes={nodes} allLeafNodes={allLeafNodes} isLeaf={isLeaf} isSelfNode={isSelfNode}
                addQuestion={addQuestion} addNode={addNode}
                onDone={() => { setSidePanel('tree'); setGraphKPSelectMode(false); setGraphKPIds([]); }}
                selectedNodeId={selectedNodeId}
                graphKPSelectMode={graphKPSelectMode}
                graphKPIds={graphKPIds}
                onEnterGraphSelect={() => setGraphKPSelectMode(true)}
                onExitGraphSelect={() => { setGraphKPSelectMode(false); setGraphKPIds([]); }} />
            )}
          </div>

          {/* Node detail */}
          {selectedNodeId && nodes[selectedNodeId] && sidePanel === 'tree' && (() => {
            const selNode = nodes[selectedNodeId];
            const selIsLeaf = isLeaf(selectedNodeId);
            const selIsExpanded = expandedNodes.has(selectedNodeId);
            const selIsSelf = isSelfNode(selectedNodeId);
            // Collapsed non-leaf or leaf: show all subtree questions
            // Expanded non-leaf: show only "self" questions (linked to node or its "XX本身" child)
            const selfChildIds = Object.values(nodes)
              .filter(n => n.parentId === selectedNodeId && isSelfNode(n.id))
              .map(n => n.id);
            const selfIds = new Set([selectedNodeId, ...selfChildIds]);
            const showAllQs = selIsLeaf || !selIsExpanded;
            const detailQs = showAllQs
              ? getNodeQuestions(selectedNodeId)
              : Object.values(questions).filter(q => q.knowledgePointIds.some(k => selfIds.has(k)));
            const detailErrors = showAllQs
              ? getNodeErrorCount(selectedNodeId)
              : detailQs.filter(q => q.status === 'wrong').length;
            return (
              <NodeDetailPanel node={selNode} isLeaf={selIsLeaf}
                mastery={showAllQs ? getNodeMastery(selectedNodeId) : mastery(detailErrors)}
                confidence={getNodeConfidence(selectedNodeId)}
                errorCount={detailErrors}
                questions={detailQs}
                allNodes={nodes}
                isExpanded={selIsExpanded}
                onDeleteNode={() => { deleteNode(selectedNodeId); setSelectedNodeId(null); }}
                onDeleteQuestion={deleteQuestion} />
            );
          })()}
        </div>

        {/* Main Graph Area */}
        <div style={styles.graphArea}>
          {view === 'graph' && (
            <ForceGraph data={graphData} selectedNodeId={selectedNodeId}
              questions={questions} nodes={nodes}
              onNodeClick={(id) => { if (!graphKPSelectMode) { setSelectedNodeId(id); setSidePanel('tree'); } }}
              onNodeDblClick={(id) => {
                setExpandedNodes(prev => {
                  const n = new Set(prev);
                  if (n.has(id)) {
                    // Collapse: remove self and ALL descendants
                    n.delete(id);
                    const removeDescendants = (nid) => {
                      getChildren(nid).forEach(c => {
                        n.delete(c.id);
                        removeDescendants(c.id);
                      });
                    };
                    removeDescendants(id);
                  } else {
                    n.add(id);
                  }
                  return n;
                });
              }}
              kpSelectMode={graphKPSelectMode}
              graphSelectedKPs={graphKPIds}
              onGraphKPToggle={(id) => setGraphKPIds(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id])}
              onAnswerQuestion={answerQuestion}
              onDeleteQuestion={deleteQuestion}
              onDeleteNode={deleteNode}
              onAddChild={(parentId) => {
                setSelectedNodeId(parentId);
                setSidePanel('addNode');
              }}
              onAddQuestion={(nodeId) => {
                setSelectedNodeId(nodeId);
                setSidePanel('addQuestion');
                setGraphKPSelectMode(true);
                setGraphKPIds(prev => [...new Set([...prev, nodeId])]);
              }} />
          )}
          {view === 'review' && (
            <ReviewPanel question={currentReviewQ} questions={questions}
              reviewIdx={reviewIdx} total={reviewQueue.length}
              nodes={nodes} allLeafNodes={allLeafNodes} isLeaf={isLeaf}
              failedKPs={failedKPs} setFailedKPs={setFailedKPs}
              reviewResult={reviewResult} setReviewResult={setReviewResult}
              onAnswer={(isCorrect) => {
                if (currentReviewQ) {
                  answerQuestion(currentReviewQ.id, isCorrect, isCorrect ? [] : failedKPs);
                }
                setReviewResult(isCorrect ? 'correct' : 'wrong');
              }}
              onNext={() => {
                setReviewResult(null); setFailedKPs([]);
                if (reviewIdx + 1 < reviewQueue.length) {
                  setReviewIdx(reviewIdx + 1);
                } else {
                  setView('graph');
                }
              }}
              onExit={() => { setView('graph'); setReviewResult(null); setFailedKPs([]); }}
            />
          )}
          {view === 'graph' && Object.keys(nodes).length === 0 && (
            <div style={styles.emptyState}>
              <div style={styles.emptyIcon}>◇</div>
              <div style={styles.emptyTitle}>开始构建你的知识图谱</div>
              <div style={styles.emptyHint}>点击左侧「+ 知识点」添加第一个章节</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ============ TREE VIEW ============
function TreeView({ nodes, rootNodes, getChildren, isLeaf, isSelfNode, expandedNodes, setExpandedNodes,
  selectedNodeId, setSelectedNodeId, getNodeMastery, getNodeConfidence, getNodeErrorCount, 
  getNodeQuestions, deleteNode }) {
  
  const renderNode = (node, depth) => {
    // Hide "XX本身" abstract nodes
    if (isSelfNode(node.id)) return null;
    const children = getChildren(node.id).filter(c => !isSelfNode(c.id));
    const expanded = expandedNodes.has(node.id);
    const selected = selectedNodeId === node.id;
    const m = getNodeMastery(node.id);
    const errors = getNodeErrorCount(node.id);
    const leaf = children.length === 0;

    return (
      <div key={node.id}>
        <div style={{
          ...styles.treeItem,
          paddingLeft: 12 + depth * 16,
          background: selected ? 'rgba(99,140,255,0.15)' : 'transparent',
          borderLeft: selected ? '3px solid #638cff' : '3px solid transparent',
        }}
          onClick={() => setSelectedNodeId(node.id)}
        >
          {!leaf && (
            <span style={styles.treeToggle} onClick={(e) => {
              e.stopPropagation();
              setExpandedNodes(prev => {
                const n = new Set(prev);
                if (n.has(node.id)) {
                  n.delete(node.id);
                  const removeAll = (nid) => { getChildren(nid).forEach(c => { n.delete(c.id); removeAll(c.id); }); };
                  removeAll(node.id);
                } else {
                  n.add(node.id);
                }
                return n;
              });
            }}>{expanded ? '▾' : '▸'}</span>
          )}
          {leaf && <span style={styles.treeLeafDot}>●</span>}
          <span style={{ ...styles.treeMasteryDot, background: masteryColor(m) }} />
          <span style={styles.treeNodeName}>{node.name}</span>
          {errors > 0 && <span style={styles.treeErrorBadge}>{errors}</span>}
        </div>
        {expanded && children.map(c => renderNode(c, depth + 1))}
      </div>
    );
  };

  return (
    <div style={styles.treeContainer}>
      {rootNodes.length === 0 && (
        <div style={styles.treeEmpty}>暂无知识点，请先添加</div>
      )}
      {rootNodes.map(r => renderNode(r, 0))}
    </div>
  );
}

// ============ ADD NODE PANEL ============
function AddNodePanel({ nodes, rootNodes, getChildren, expandedNodes, setExpandedNodes, addNode, onDone, selectedNodeId }) {
  const [parentId, setParentId] = useState(selectedNodeId || null);
  const [name, setName] = useState('');

  // Update parentId when selectedNodeId changes
  useEffect(() => { if (selectedNodeId) setParentId(selectedNodeId); }, [selectedNodeId]);

  const handleAdd = () => {
    if (!name.trim()) return;
    addNode(parentId, name.trim());
    setName('');
  };

  return (
    <div style={styles.formPanel}>
      <div style={styles.formLabel}>父节点（留空则创建顶级章节）</div>
      <select style={styles.formSelect} value={parentId || ''} onChange={e => setParentId(e.target.value || null)}>
        <option value="">无（顶级章节）</option>
        {Object.values(nodes).sort((a,b) => a.name.localeCompare(b.name)).map(n => (
          <option key={n.id} value={n.id}>
            {getNodePath(nodes, n.id)}
          </option>
        ))}
      </select>
      <div style={styles.formLabel}>知识点名称</div>
      <input style={styles.formInput} value={name} onChange={e => setName(e.target.value)}
        placeholder="如：多元函数微分学" onKeyDown={e => e.key === 'Enter' && handleAdd()} />
      <button style={styles.formBtn} onClick={handleAdd} disabled={!name.trim()}>添加知识点</button>
    </div>
  );
}

function getNodePath(nodes, id) {
  const parts = [];
  let cur = id;
  while (cur && nodes[cur]) {
    parts.unshift(nodes[cur].name);
    cur = nodes[cur].parentId;
  }
  return parts.join(' / ');
}

// ============ ADD QUESTION PANEL ============
function AddQuestionPanel({ nodes, allLeafNodes, isLeaf, isSelfNode, addQuestion, addNode, onDone, selectedNodeId, graphKPSelectMode, graphKPIds, onEnterGraphSelect, onExitGraphSelect }) {
  const [content, setContent] = useState('');
  const [selectedKPs, setSelectedKPs] = useState(selectedNodeId ? [selectedNodeId] : []);
  const [difficulty, setDifficulty] = useState(3);
  const [status, setStatus] = useState('wrong');
  const [failedKPs, setFailedKPs] = useState([]);
  const [search, setSearch] = useState('');
  const [treeExpanded, setTreeExpanded] = useState(new Set(
    selectedNodeId ? getAncestorIds(nodes, selectedNodeId) : []
  ));

  // Sync KPs selected on graph into local selection
  useEffect(() => {
    if (graphKPIds && graphKPIds.length > 0) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setSelectedKPs(prev => [...new Set([...prev, ...graphKPIds])]);
    }
  }, [graphKPIds]);

  const handleSubmit = () => {
    if (!content.trim() || selectedKPs.length === 0) return;
    let kpIds = [...selectedKPs];
    kpIds = kpIds.map(kpId => {
      if (!isLeaf(kpId)) {
        const name = nodes[kpId].name + '本身';
        const existing = Object.values(nodes).find(n => n.parentId === kpId && n.name === name);
        if (existing) return existing.id;
        return addNode(kpId, name);
      }
      return kpId;
    });
    const fkp = status === 'wrong' ? (failedKPs.length > 0 ? failedKPs : kpIds) : [];
    addQuestion(content.trim(), kpIds, difficulty, status, fkp);
    setContent(''); setSelectedKPs([]); setFailedKPs([]);
  };

  const rootNodes = Object.values(nodes).filter(n => !n.parentId && !isSelfNode(n.id)).sort((a,b) => a.name.localeCompare(b.name));
  const getChildren = (id) => Object.values(nodes).filter(n => n.parentId === id && !isSelfNode(n.id)).sort((a,b) => a.name.localeCompare(b.name));

  const matchesSearch = (id) => {
    if (!search) return true;
    const s = search.toLowerCase();
    if (nodes[id].name.toLowerCase().includes(s)) return true;
    return getChildren(id).some(c => matchesSearch(c.id));
  };

  const renderTreeNode = (node, depth) => {
    if (!matchesSearch(node.id)) return null;
    const children = getChildren(node.id);
    const expanded = treeExpanded.has(node.id) || (search && matchesSearch(node.id));
    const leaf = children.length === 0;
    const checked = selectedKPs.includes(node.id);

    return (
      <div key={node.id}>
        <div style={{ display: 'flex', alignItems: 'center', padding: '3px 0', paddingLeft: depth * 14 }}>
          {!leaf && (
            <span style={{ cursor: 'pointer', fontSize: 10, color: '#8b949e', width: 14, textAlign: 'center', flexShrink: 0 }}
              onClick={() => setTreeExpanded(prev => {
                const n = new Set(prev);
                n.has(node.id) ? n.delete(node.id) : n.add(node.id);
                return n;
              })}>{expanded ? '▾' : '▸'}</span>
          )}
          {leaf && <span style={{ width: 14, flexShrink: 0 }} />}
          <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', flex: 1, minWidth: 0 }}>
            <input type="checkbox" checked={checked}
              onChange={e => setSelectedKPs(prev => e.target.checked ? [...prev, node.id] : prev.filter(x => x !== node.id))}
              style={{ flexShrink: 0 }} />
            <span style={{ marginLeft: 4, fontSize: 12, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              color: checked ? '#638cff' : '#c8d0e0' }}>
              {node.name}
            </span>
          </label>
        </div>
        {(expanded || search) && children.map(c => renderTreeNode(c, depth + 1))}
      </div>
    );
  };

  return (
    <div style={styles.formPanel}>
      <div style={styles.formLabel}>题目内容</div>
      <textarea style={styles.formTextarea} value={content} onChange={e => setContent(e.target.value)}
        placeholder="题干或标识，如：660第15题" rows={3} />
      
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: 4 }}>
        <div style={styles.formLabel}>关联知识点</div>
        {!graphKPSelectMode ? (
          <button style={{ background: 'rgba(96,216,255,0.08)', border: '1px solid rgba(96,216,255,0.25)', color: '#60d8ff', fontSize: 11, padding: '2px 10px', borderRadius: 8, cursor: 'pointer' }}
            onClick={onEnterGraphSelect}>在图中选择</button>
        ) : (
          <button style={{ background: 'rgba(63,185,80,0.12)', border: '1px solid rgba(63,185,80,0.35)', color: '#3fb950', fontSize: 11, padding: '2px 10px', borderRadius: 8, cursor: 'pointer', fontWeight: 600 }}
            onClick={onExitGraphSelect}>✓ 完成选择</button>
        )}
      </div>
      {graphKPSelectMode && (
        <div style={{ fontSize: 11, color: '#60d8ff', background: 'rgba(96,216,255,0.06)', border: '1px solid rgba(96,216,255,0.2)', borderRadius: 6, padding: '5px 10px' }}>
          在右侧图谱上点击知识点节点来添加关联
        </div>
      )}
      <input style={{ ...styles.formInput, marginBottom: 6 }} value={search}
        onChange={e => setSearch(e.target.value)} placeholder="搜索知识点..." />
      <div style={styles.kpSelector}>
        {rootNodes.map(r => renderTreeNode(r, 0))}
        {rootNodes.length === 0 && <div style={{ color: '#484f58', fontSize: 12, padding: 8 }}>暂无知识点</div>}
      </div>
      {selectedKPs.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 4 }}>
          {selectedKPs.map(kpId => (
            <span key={kpId} style={styles.kpTag}>
              {nodes[kpId]?.name}
              <span style={styles.kpTagX} onClick={() => setSelectedKPs(prev => prev.filter(x => x !== kpId))}>×</span>
            </span>
          ))}
        </div>
      )}

      <div style={styles.formLabel}>难度</div>
      <div style={styles.difficultyRow}>
        {[1,2,3,4,5].map(d => (
          <button key={d} style={difficulty === d ? styles.diffBtnActive : styles.diffBtn}
            onClick={() => setDifficulty(d)}>{d}</button>
        ))}
      </div>

      <div style={styles.formLabel}>当前状态</div>
      <div style={styles.statusRow}>
        <button style={status === 'correct' ? styles.statusBtnCorrect : styles.statusBtn}
          onClick={() => setStatus('correct')}>✓ 做对了</button>
        <button style={status === 'wrong' ? styles.statusBtnWrong : styles.statusBtn}
          onClick={() => setStatus('wrong')}>✗ 做错了</button>
        <button style={status === 'untested' ? styles.statusBtnActive : styles.statusBtn}
          onClick={() => setStatus('untested')}>未测试</button>
      </div>

      {status === 'wrong' && selectedKPs.length > 1 && (
        <>
          <div style={styles.formLabel}>因为哪些知识点不会？（不选则默认全部）</div>
          <div style={styles.kpSelector}>
            {selectedKPs.map(kpId => (
              <label key={kpId} style={styles.kpOption}>
                <input type="checkbox" checked={failedKPs.includes(kpId)}
                  onChange={e => setFailedKPs(prev => e.target.checked ? [...prev, kpId] : prev.filter(x => x !== kpId))} />
                <span style={{ marginLeft: 6, fontSize: 13 }}>{nodes[kpId]?.name || kpId}</span>
              </label>
            ))}
          </div>
        </>
      )}

      <button style={styles.formBtn} onClick={handleSubmit}
        disabled={!content.trim() || selectedKPs.length === 0}>录入题目</button>
    </div>
  );
}

function getAncestorIds(nodes, id) {
  const ids = [];
  let cur = nodes[id]?.parentId;
  while (cur && nodes[cur]) { ids.push(cur); cur = nodes[cur].parentId; }
  return ids;
}

// ============ NODE DETAIL PANEL ============
function NodeDetailPanel({ node, isLeaf, isExpanded, mastery: m, confidence, errorCount, questions, allNodes, onDeleteNode, onDeleteQuestion }) {
  const [selectedQId, setSelectedQId] = useState(null);
  const [qCtxMenu, setQCtxMenu] = useState(null);
  const selectedQ = selectedQId ? questions.find(q => q.id === selectedQId) : null;

  return (
    <div style={styles.detailPanel} onClick={() => setQCtxMenu(null)}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={styles.detailTitle}>
          {node.name}
          {!isLeaf && <span style={{ fontSize: 11, color: '#8b949e', marginLeft: 6 }}>{isExpanded ? '（本身）' : '（整棵树）'}</span>}
        </div>
        <button style={styles.deleteNodeBtn} onClick={onDeleteNode} title="删除此知识点">✕</button>
      </div>
      <div style={styles.detailPath}>{getNodePath(allNodes, node.id)}</div>
      <div style={styles.detailStats}>
        <div style={styles.detailStatRow}>
          <span>掌握度</span>
          <div style={styles.masteryBar}>
            <div style={{ ...styles.masteryBarFill, width: `${m * 100}%`, background: masteryColor(m) }} />
          </div>
          <span style={{ color: masteryColor(m), fontWeight: 600 }}>{(m * 100).toFixed(0)}%</span>
        </div>
        {isLeaf && (
          <div style={styles.detailStatRow}>
            <span>置信度</span>
            <div style={styles.masteryBar}>
              <div style={{ ...styles.masteryBarFill, width: `${confidence * 100}%`, background: '#638cff' }} />
            </div>
            <span style={{ color: '#638cff', fontWeight: 600 }}>{(confidence * 100).toFixed(0)}%</span>
          </div>
        )}
        <div style={styles.detailStatRow}>
          <span>错题数</span>
          <span style={{ color: errorCount > 0 ? '#ff6b6b' : '#6bffa0', fontWeight: 600 }}>{errorCount}</span>
        </div>
        <div style={styles.detailStatRow}>
          <span>总题数</span>
          <span style={{ fontWeight: 600 }}>{questions.length}</span>
        </div>
        {isLeaf && (
          <div style={styles.detailStatRow}>
            <span>遗忘倾向</span>
            <span style={{ fontWeight: 600 }}>{((node.forgetTendency || 0) * 100).toFixed(0)}%</span>
          </div>
        )}
      </div>
      {questions.length > 0 && (
        <div style={styles.detailQList}>
          <div style={styles.detailQListTitle}>题目列表</div>
          {questions.slice(0, 50).map(q => (
            <div key={q.id} style={{ position: 'relative' }}>
              <div style={{
                ...styles.detailQItem,
                background: selectedQId === q.id ? 'rgba(99,140,255,0.1)' : 'transparent',
                cursor: 'pointer',
              }}
                onClick={() => setSelectedQId(selectedQId === q.id ? null : q.id)}
                onContextMenu={(e) => {
                  e.preventDefault(); e.stopPropagation();
                  setQCtxMenu({ qId: q.id, x: e.clientX, y: e.clientY });
                }}
              >
                <span style={{
                  ...styles.qStatusDot,
                  background: q.status === 'correct' ? '#6bffa0' : q.status === 'wrong' ? '#ff6b6b' : '#888'
                }} />
                <span style={styles.qContent}>{q.content}</span>
                <span style={styles.qDiff}>{'★'.repeat(q.difficulty)}</span>
              </div>
              {selectedQId === q.id && (
                <div style={styles.qDetail}>
                  <div style={styles.qDetailRow}>状态：<span style={{ color: q.status === 'wrong' ? '#ff7b72' : '#3fb950' }}>{q.status === 'wrong' ? '错误' : q.status === 'correct' ? '正确' : '未测试'}</span></div>
                  <div style={styles.qDetailRow}>难度：{'★'.repeat(q.difficulty)}{'☆'.repeat(5 - q.difficulty)}</div>
                  <div style={styles.qDetailRow}>关联：{q.knowledgePointIds.map(k => allNodes[k]?.name || '?').join('、')}</div>
                  {q.failedKnowledgePointIds?.length > 0 && (
                    <div style={styles.qDetailRow}>失败原因：{q.failedKnowledgePointIds.map(k => allNodes[k]?.name || '?').join('、')}</div>
                  )}
                  <div style={styles.qDetailRow}>做题次数：{q.history?.length || 0}</div>
                  <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
                    <button style={{ ...styles.deleteQBtn, color: '#ff7b72', fontSize: 12, padding: '2px 8px', border: '1px solid #30363d', borderRadius: 4 }}
                      onClick={(e) => { e.stopPropagation(); onDeleteQuestion(q.id); setSelectedQId(null); }}>删除</button>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
      {qCtxMenu && (
        <div style={{
          position: 'fixed', left: qCtxMenu.x, top: qCtxMenu.y,
          background: '#21262d', border: '1px solid #30363d', borderRadius: 6,
          padding: 4, zIndex: 200, boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
        }}>
          <button style={styles.ctxMenuItem} onClick={() => { setSelectedQId(qCtxMenu.qId); setQCtxMenu(null); }}>
            查看详情
          </button>
          <button style={{ ...styles.ctxMenuItem, color: '#ff7b72' }} onClick={() => {
            onDeleteQuestion(qCtxMenu.qId); setQCtxMenu(null); setSelectedQId(null);
          }}>
            删除题目
          </button>
        </div>
      )}
    </div>
  );
}

// ============ FORCE GRAPH ============
function ForceGraph({ data, selectedNodeId, onNodeClick, onNodeDblClick, onDeleteNode, onAddChild, onAddQuestion, questions, nodes, kpSelectMode, graphSelectedKPs, onGraphKPToggle, onAnswerQuestion, onDeleteQuestion }) {
  const svgRef = useRef(null);
  const simRef = useRef(null);
  const gRef = useRef(null);
  const ghostGRef = useRef(null);
  const ghostSimRef = useRef(null);
  const posRef = useRef({});
  const zoomRef = useRef(null);
  const zoomTransformRef = useRef(null);
  const selectedRef = useRef(selectedNodeId);
  const showQRef = useRef(null);
  const dataRef = useRef(data);
  const questionsRef = useRef(questions);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [contextMenu, setContextMenu] = useState(null);
  const [showQuestionsFor, setShowQuestionsFor] = useState(null);
  const [tooltipInfo, setTooltipInfo] = useState(null);
  const [detailQ, setDetailQ] = useState(null);
  const [answeringDetailQ, setAnsweringDetailQ] = useState(false);
  const [detailQFailedKPs, setDetailQFailedKPs] = useState([]);
  const [detailQAnswerResult, setDetailQAnswerResult] = useState(null);
  selectedRef.current = selectedNodeId;
  dataRef.current = data;
  questionsRef.current = questions;
  showQRef.current = showQuestionsFor;
  const nodesTreeRef = useRef(nodes);
  nodesTreeRef.current = nodes;
  const kpSelectModeRef = useRef(kpSelectMode);
  kpSelectModeRef.current = kpSelectMode;
  const onGraphKPToggleRef = useRef(onGraphKPToggle);
  onGraphKPToggleRef.current = onGraphKPToggle;

  useEffect(() => {
    const container = svgRef.current?.parentElement;
    if (container) {
      const ro = new ResizeObserver(entries => {
        const { width, height } = entries[0].contentRect;
        setDimensions({ width, height });
      });
      ro.observe(container);
      return () => ro.disconnect();
    }
  }, []);

  useEffect(() => {
    if (!gRef.current) return;
    d3.select(gRef.current).selectAll('circle.main-circle')
      .attr('stroke', d => {
        if (d.id === selectedNodeId) return '#fff';
        if (graphSelectedKPs?.includes(d.id)) return '#60d8ff';
        return 'rgba(255,255,255,0.15)';
      })
      .attr('stroke-width', d => (d.id === selectedNodeId || graphSelectedKPs?.includes(d.id)) ? 3 : 1);
  }, [selectedNodeId, graphSelectedKPs]);

  // Ghost system
  useEffect(() => {
    if (ghostSimRef.current) { ghostSimRef.current.stop(); ghostSimRef.current = null; }
    if (!ghostGRef.current) return;
    const ghostG = d3.select(ghostGRef.current);
    ghostG.selectAll('*').remove();
    const sqf = showQRef.current;
    if (!sqf) return;
    const curData = dataRef.current;
    const curQuestions = questionsRef.current;
    const nodeData = curData.nodes.find(n => n.id === sqf);
    if (!nodeData || !nodeData.x) return;
    const va = curData.visibleAncestor || {};
    const allTreeNodes = nodesTreeRef.current;
    const showSelfOnly = nodeData.isExpanded && !nodeData.isLeaf;
    let myIds;
    if (showSelfOnly) {
      myIds = new Set([sqf]);
      Object.values(allTreeNodes).forEach(n => {
        if (n.parentId === sqf && n.name === allTreeNodes[sqf]?.name + '本身') myIds.add(n.id);
      });
    } else {
      myIds = new Set([sqf]);
      const collect = (nid) => { Object.values(allTreeNodes).forEach(n => { if (n.parentId === nid) { myIds.add(n.id); collect(n.id); } }); };
      collect(sqf);
    }
    const qs = Object.values(curQuestions || {}).filter(q => q.knowledgePointIds.some(k => myIds.has(k)));
    if (qs.length === 0) return;

    // Split into self-only and cross-KP
    const selfQs = [];
    const crossGroups = {};
    qs.forEach(q => {
      const ext = q.knowledgePointIds.filter(k => !myIds.has(k));
      if (ext.length === 0) { selfQs.push(q); } else {
        const resolved = [...new Set(ext.map(k => va[k] || k).filter(Boolean))];
        resolved.forEach(tid => {
          if (tid === sqf) return;
          if (!crossGroups[tid]) crossGroups[tid] = { questions: [], targetNode: curData.nodes.find(n => n.id === tid), tid };
          crossGroups[tid].questions.push(q);
        });
      }
    });
    const crossGroupsArr = Object.values(crossGroups).filter(cg => cg.targetNode?.x);

    // ---- Self-question simulation (clustered balls near KP node) ----
    const gNodes = [];
    const gLinks = [];
    const anchor = { id: '_anchor', fx: nodeData.x, fy: nodeData.y, isAnchor: true, radius: 0 };
    gNodes.push(anchor);
    selfQs.forEach((q) => {
      const gn = {
        id: '_sq_' + q.id,
        x: nodeData.x + (Math.random() - 0.5) * 12,
        y: nodeData.y + (Math.random() - 0.5) * 12,
        radius: 4, isQuestion: true, status: q.status, questionData: q,
      };
      gNodes.push(gn);
      gLinks.push({ source: '_anchor', target: gn.id, distance: nodeData.radius + 8, strength: 0.9 });
    });

    const gSim = d3.forceSimulation(gNodes)
      .force('link', d3.forceLink(gLinks).id(d => d.id)
        .distance(d => d.distance || 20).strength(d => d.strength != null ? d.strength : 0.6))
      .force('charge', d3.forceManyBody().strength(0))
      .force('collision', d3.forceCollide(d => d.isQuestion ? d.radius + 1 : d.radius + 2).strength(0.9))
      .force('cluster', () => {
        gNodes.forEach(n => {
          if (!n.isQuestion) return;
          n.vx += (anchor.fx + nodeData.radius + 16 - n.x) * 0.08;
          n.vy += (anchor.fy + 10 - n.y) * 0.05;
        });
      })
      .velocityDecay(0.72).alpha(0.5).alphaDecay(0.02);
    ghostSimRef.current = gSim;

    // ---- Render layers (ghostLinks behind, ghostNodeG in front) ----
    const ghostLinks = ghostG.append('g');
    const ghostNodeG = ghostG.append('g');

    // Self-question lines
    ghostLinks.selectAll('line.qline').data(gLinks).join('line').attr('class', 'qline')
      .attr('stroke', 'rgba(180,200,255,0.2)').attr('stroke-width', 0.8);

    // Self-question balls with hover tooltip + click detail
    const qBallG = ghostNodeG.selectAll('g.qball')
      .data(gNodes.filter(d => d.isQuestion), d => d.id)
      .join('g').attr('class', 'qball').style('cursor', 'pointer');
    qBallG.append('circle').attr('r', d => d.radius)
      .attr('fill', d => d.status === 'wrong' ? 'rgba(255,100,100,0.6)' : 'rgba(100,220,150,0.5)')
      .attr('stroke', 'rgba(255,255,255,0.25)').attr('stroke-width', 0.5);
    qBallG
      .on('mouseover', (e, d) => {
        const rect = svgRef.current.getBoundingClientRect();
        setTooltipInfo({ x: e.clientX - rect.left, y: e.clientY - rect.top, q: d.questionData });
      })
      .on('mousemove', (e) => {
        const rect = svgRef.current.getBoundingClientRect();
        setTooltipInfo(prev => prev ? { ...prev, x: e.clientX - rect.left, y: e.clientY - rect.top } : null);
      })
      .on('mouseleave', () => setTooltipInfo(null))
      .on('click', (e, d) => {
        e.stopPropagation();
        setDetailQ(prev => prev?.id === d.questionData?.id ? null : d.questionData);
      });

    // ---- Cross-KP: badge collapses to single line, expands to fan on the line ----
    const MAX_VISIBLE = 8;
    const SPACING = 14;
    const expandedCross = new Set();
    const cgRefs = [];

    crossGroupsArr.forEach(cg => {
      const displayQs = cg.questions.slice(0, MAX_VISIBLE);
      const extraCount = cg.questions.length - displayQs.length;

      // Lines layer
      const lineG = ghostLinks.append('g').attr('class', 'cg-linegroup').datum(cg);
      lineG.append('line').attr('class', 'cg-main-line')
        .attr('stroke', 'rgba(255,200,100,0.35)').attr('stroke-width', 2.5).attr('stroke-dasharray', '5,3');
      const fanLinesG = lineG.append('g').attr('class', 'cg-fanlines').style('display', 'none');
      displayQs.forEach((q, i) => {
        const lc = q.status === 'wrong' ? 'rgba(255,130,130,0.5)' : 'rgba(130,220,160,0.5)';
        const fi = fanLinesG.append('g').attr('class', 'cg-fli').datum({ q, i, N: displayQs.length });
        fi.append('line').attr('class', 'cg-fl1').attr('stroke', lc).attr('stroke-width', 1.2);
        fi.append('line').attr('class', 'cg-fl2').attr('stroke', lc).attr('stroke-width', 1.2);
      });

      // Nodes layer
      const nodeGEl = ghostNodeG.append('g').attr('class', 'cg-nodegroup').datum(cg);
      // Collapsed badge
      const badgeG = nodeGEl.append('g').attr('class', 'cg-badge').style('cursor', 'pointer');
      badgeG.append('rect').attr('x', -18).attr('y', -11).attr('width', 36).attr('height', 22)
        .attr('rx', 5).attr('fill', 'rgba(255,200,100,0.2)').attr('stroke', 'rgba(255,200,100,0.6)').attr('stroke-width', 1.2);
      badgeG.append('text').attr('text-anchor', 'middle').attr('dy', 4)
        .attr('fill', 'rgba(255,215,120,0.9)').attr('font-size', 10).attr('font-family', 'system-ui')
        .text(cg.questions.length + '题').style('pointer-events', 'none');
      // Expanded fan balls
      const fanBallsG = nodeGEl.append('g').attr('class', 'cg-fanballs').style('display', 'none');
      displayQs.forEach((q, i) => {
        fanBallsG.append('g').attr('class', 'cg-fbi').datum({ q, i, N: displayQs.length })
          .style('cursor', 'pointer')
          .append('circle').attr('r', 5)
          .attr('fill', q.status === 'wrong' ? 'rgba(255,100,100,0.75)' : 'rgba(100,220,150,0.7)')
          .attr('stroke', 'rgba(255,255,255,0.35)').attr('stroke-width', 0.8);
      });
      if (extraCount > 0) {
        const moreG = fanBallsG.append('g').attr('class', 'cg-fanmore');
        moreG.append('circle').attr('r', 9)
          .attr('fill', 'rgba(160,160,160,0.3)').attr('stroke', 'rgba(180,180,180,0.5)').attr('stroke-width', 1);
        moreG.append('text').attr('text-anchor', 'middle').attr('dy', 3)
          .attr('fill', 'rgba(200,200,200,0.9)').attr('font-size', 7).attr('font-family', 'system-ui')
          .text('+' + extraCount).style('pointer-events', 'none');
      }

      // Fan ball hover / click
      fanBallsG.selectAll('g.cg-fbi')
        .on('mouseover', (e, d) => {
          const rect = svgRef.current.getBoundingClientRect();
          setTooltipInfo({ x: e.clientX - rect.left, y: e.clientY - rect.top, q: d.q });
        })
        .on('mousemove', (e) => {
          const rect = svgRef.current.getBoundingClientRect();
          setTooltipInfo(prev => prev ? { ...prev, x: e.clientX - rect.left, y: e.clientY - rect.top } : null);
        })
        .on('mouseleave', () => setTooltipInfo(null))
        .on('click', (e, d) => {
          e.stopPropagation();
          setDetailQ(prev => prev?.id === d.q.id ? null : d.q);
        });

      // Badge click: toggle fan expand / collapse
      badgeG.on('click', (e) => {
        e.stopPropagation();
        if (expandedCross.has(cg.tid)) {
          expandedCross.delete(cg.tid);
          lineG.select('.cg-main-line').style('display', null);
          lineG.select('.cg-fanlines').style('display', 'none');
          badgeG.style('display', null);
          fanBallsG.style('display', 'none');
        } else {
          expandedCross.add(cg.tid);
          lineG.select('.cg-main-line').style('display', 'none');
          lineG.select('.cg-fanlines').style('display', null);
          badgeG.style('display', 'none');
          fanBallsG.style('display', null);
        }
      });

      cgRefs.push({ cg, lineG, nodeGEl, displayQs, extraCount });
    });

    // Tick: physics for self-balls + pure geometry for cross-KP fan
    gSim.on('tick', () => {
      const nd = dataRef.current.nodes.find(n => n.id === sqf);
      if (nd) { anchor.fx = nd.x; anchor.fy = nd.y; }

      ghostNodeG.selectAll('g.qball').attr('transform', d => `translate(${d.x},${d.y})`);
      ghostLinks.selectAll('line.qline').each(function(d) {
        const src = d.source; const tgt = d.target;
        if (src && tgt) d3.select(this).attr('x1', src.x || 0).attr('y1', src.y || 0).attr('x2', tgt.x || 0).attr('y2', tgt.y || 0);
      });

      // Recompute cross-KP geometry every tick (tracks moving KP nodes)
      cgRefs.forEach(({ cg, lineG, nodeGEl, displayQs, extraCount }) => {
        const sn = dataRef.current.nodes.find(n => n.id === sqf);
        const tn = dataRef.current.nodes.find(n => n.id === cg.tid);
        if (!sn || !tn) return;
        const mx = (sn.x + tn.x) / 2;
        const my = (sn.y + tn.y) / 2;

        if (!expandedCross.has(cg.tid)) {
          lineG.select('.cg-main-line')
            .attr('x1', sn.x).attr('y1', sn.y).attr('x2', tn.x).attr('y2', tn.y);
          nodeGEl.select('.cg-badge').attr('transform', `translate(${mx},${my})`);
        } else {
          const dx = tn.x - sn.x;
          const dy = tn.y - sn.y;
          const len = Math.sqrt(dx * dx + dy * dy) || 1;
          const px = -dy / len;  // perpendicular unit vector
          const py = dx / len;
          const N = displayQs.length;

          lineG.selectAll('g.cg-fli').each(function(d) {
            const offset = (d.i - (N - 1) / 2) * SPACING;
            const bx = mx + px * offset;
            const by = my + py * offset;
            d3.select(this).select('.cg-fl1').attr('x1', sn.x).attr('y1', sn.y).attr('x2', bx).attr('y2', by);
            d3.select(this).select('.cg-fl2').attr('x1', bx).attr('y1', by).attr('x2', tn.x).attr('y2', tn.y);
          });

          nodeGEl.selectAll('g.cg-fbi').each(function(d) {
            const offset = (d.i - (N - 1) / 2) * SPACING;
            d3.select(this).attr('transform', `translate(${mx + px * offset},${my + py * offset})`);
          });

          if (extraCount > 0) {
            const offset = ((N - 1) / 2 + 1) * SPACING + 5;
            nodeGEl.select('.cg-fanmore')
              .attr('transform', `translate(${mx + px * offset},${my + py * offset})`);
          }
        }
      });
    });

    return () => { if (ghostSimRef.current) ghostSimRef.current.stop(); };
  }, [showQuestionsFor]);

  // Reheat ghost sim when main sim ticks (keep anchors tracking)
  const reheatGhost = useCallback(() => {
    if (ghostSimRef.current && showQRef.current) {
      ghostSimRef.current.alpha(Math.max(ghostSimRef.current.alpha(), 0.02)).restart();
    }
  }, []);

  // Main simulation
  useEffect(() => {
    if (!svgRef.current || data.nodes.length === 0) {
      const svg = d3.select(svgRef.current);
      if (svg) svg.selectAll('*').remove();
      gRef.current = null; ghostGRef.current = null;
      return;
    }
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    setContextMenu(null);
    const { width, height } = dimensions;

    // Defs for gradients
    const defs = svg.append('defs');
    const g = svg.append('g');
    gRef.current = g.node();

    const zoom = d3.zoom().scaleExtent([0.2, 4]).on('zoom', (e) => { g.attr('transform', e.transform); zoomTransformRef.current = e.transform; });
    svg.call(zoom);
    if (zoomTransformRef.current) svg.call(zoom.transform, zoomTransformRef.current);
    zoomRef.current = zoom;
    svg.on('click', () => setContextMenu(null));

    const hasPositions = data.nodes.some(n => posRef.current[n.id]);
    const parentMap = {};
    data.links.forEach(l => {
      if (l.type === 'parent') {
        parentMap[typeof l.target === 'object' ? l.target.id : l.target] = typeof l.source === 'object' ? l.source.id : l.source;
      }
    });
    data.nodes.forEach(n => {
      if (posRef.current[n.id]) { n.x = posRef.current[n.id].x; n.y = posRef.current[n.id].y; }
      else if (parentMap[n.id] && posRef.current[parentMap[n.id]]) {
        const pp = posRef.current[parentMap[n.id]];
        const a = Math.random() * Math.PI * 2;
        n.x = pp.x + Math.cos(a) * (40 + Math.random() * 30);
        n.y = pp.y + Math.sin(a) * (40 + Math.random() * 30);
      } else { n.x = width/2 + (Math.random()-0.5)*200; n.y = height/2 + (Math.random()-0.5)*200; }
    });

    // Create gradients for parent links
    const parentLinks = data.links.filter(l => l.type === 'parent');
    parentLinks.forEach((l, i) => {
      const grad = defs.append('linearGradient').attr('id', `pgrad-${i}`).attr('gradientUnits', 'userSpaceOnUse');
      grad.append('stop').attr('offset', '0%').attr('stop-color', 'rgba(160,195,255,0.95)');
      grad.append('stop').attr('offset', '100%').attr('stop-color', 'rgba(30,40,70,0.08)');
      l._gradId = `pgrad-${i}`;
    });

    const sim = d3.forceSimulation(data.nodes)
      .force('link', d3.forceLink(data.links).id(d => d.id).distance(d => d.type === 'parent' ? 85 : 160).strength(d => d.type === 'parent' ? 0.6 : 0.03))
      .force('charge', d3.forceManyBody().strength(-100).distanceMax(200))
      .force('collision', d3.forceCollide().radius(d => d.radius + 6).strength(0.7))
      .velocityDecay(0.65);
    if (!hasPositions) { sim.force('center', d3.forceCenter(width/2, height/2).strength(0.05)); sim.alpha(0.5).alphaDecay(0.03); }
    else { sim.alpha(0.08).alphaDecay(0.04); }
    simRef.current = sim;

    // Links
    const linkG = g.append('g');
    const questionLinks = data.links.filter(l => l.type === 'question');
    // Question cross-links (dashed)
    linkG.selectAll('line.qlink').data(questionLinks).join('line').attr('class', 'qlink')
      .attr('stroke', 'rgba(255,200,100,0.12)').attr('stroke-width', d => Math.min((d.count||1)*0.8, 4)).attr('stroke-dasharray', '4,4');
    // Parent links with gradient
    const pLinks = linkG.selectAll('line.plink').data(parentLinks).join('line').attr('class', 'plink')
      .attr('stroke', d => `url(#${d._gradId})`).attr('stroke-width', 3.5);

    // Ghost layer
    const ghostG = g.append('g');
    ghostGRef.current = ghostG.node();

    // Node groups
    const nodeG = g.append('g');
    const node = nodeG.selectAll('g').data(data.nodes).join('g').style('cursor', 'pointer')
      .call(d3.drag()
        .on('start', (e, d) => { d.fx = d.x; d.fy = d.y; })
        .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; sim.alpha(0.03).restart(); })
        .on('end', (e, d) => { d.fx = null; d.fy = null; })
      );

    node.append('circle').attr('class', 'main-circle').attr('r', d => d.radius)
      .attr('fill', d => masteryColor(d.mastery))
      .attr('stroke', d => d.id === selectedRef.current ? '#fff' : 'rgba(255,255,255,0.15)')
      .attr('stroke-width', d => d.id === selectedRef.current ? 3 : 1).attr('opacity', 0.9);
    node.filter(d => d.mastery < 0.2 && d.errorCount > 0).append('circle')
      .attr('r', d => d.radius + 6).attr('fill', 'none').attr('stroke', '#ff4444')
      .attr('stroke-width', 2).attr('opacity', 0.6).attr('stroke-dasharray', '3,3');
    node.append('text').text(d => d.name.length > 10 ? d.name.slice(0,9)+'…' : d.name)
      .attr('text-anchor', 'middle').attr('dy', d => d.radius + 14)
      .attr('fill', '#c8d0e0').attr('font-size', 11).attr('font-family', 'system-ui, sans-serif').style('pointer-events', 'none');
    node.filter(d => d.errorCount > 0).append('text').text(d => d.errorCount)
      .attr('text-anchor', 'middle').attr('dy', 4).attr('fill', '#fff')
      .attr('font-size', d => Math.min(d.radius * 0.8, 14)).attr('font-weight', 'bold')
      .attr('font-family', 'system-ui, sans-serif').style('pointer-events', 'none');

    node.on('click', (e, d) => {
      e.stopPropagation();
      setContextMenu(null);
      if (kpSelectModeRef.current) {
        onGraphKPToggleRef.current(d.id);
      } else {
        onNodeClick(d.id);
      }
    });
    node.on('dblclick', (e, d) => { e.stopPropagation(); onNodeDblClick(d.id); });
    node.on('contextmenu', (e, d) => {
      e.preventDefault(); e.stopPropagation();
      const rect = svgRef.current.getBoundingClientRect();
      setContextMenu({ x: e.clientX - rect.left, y: e.clientY - rect.top, nodeId: d.id, nodeName: d.name });
    });

    sim.on('tick', () => {
      // Update all links
      linkG.selectAll('line.qlink').attr('x1', d => d.source.x).attr('y1', d => d.source.y).attr('x2', d => d.target.x).attr('y2', d => d.target.y);
      pLinks.attr('x1', d => d.source.x).attr('y1', d => d.source.y).attr('x2', d => d.target.x).attr('y2', d => d.target.y);
      // Update gradient coordinates
      parentLinks.forEach((l, i) => {
        defs.select(`#pgrad-${i}`)
          .attr('x1', l.source.x).attr('y1', l.source.y)
          .attr('x2', l.target.x).attr('y2', l.target.y);
      });
      node.attr('transform', d => `translate(${d.x},${d.y})`);
      data.nodes.forEach(d => { posRef.current[d.id] = { x: d.x, y: d.y }; });
      reheatGhost();
    });

    return () => sim.stop();
  }, [data, dimensions]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}
      onClick={() => { setDetailQ(null); setContextMenu(null); }}
      onContextMenu={e => e.preventDefault()}>
      <svg ref={svgRef} width={dimensions.width} height={dimensions.height} style={{ background: '#0d1117', borderRadius: 8 }} />

      {/* KP select mode banner */}
      {kpSelectMode && (
        <div style={{
          position: 'absolute', top: 12, left: '50%', transform: 'translateX(-50%)',
          background: 'rgba(96,216,255,0.92)', color: '#0d1117', padding: '8px 20px',
          borderRadius: 20, fontSize: 13, fontWeight: 600, zIndex: 20,
          boxShadow: '0 2px 10px rgba(0,0,0,0.4)', pointerEvents: 'none', whiteSpace: 'nowrap',
        }}>
          点击节点选择关联知识点 · 已选 {graphSelectedKPs?.length || 0} 个
        </div>
      )}
      {/* Hover tooltip for question balls */}
      {tooltipInfo && (
        <div style={{
          position: 'absolute', left: tooltipInfo.x + 14, top: tooltipInfo.y - 12,
          background: '#21262d', border: '1px solid #30363d', borderRadius: 6,
          padding: '6px 10px', maxWidth: 260, pointerEvents: 'none', zIndex: 50,
          boxShadow: '0 2px 10px rgba(0,0,0,0.5)', fontSize: 11, color: '#c8d0e0',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
            <span style={{
              width: 6, height: 6, borderRadius: '50%', flexShrink: 0, display: 'inline-block',
              background: tooltipInfo.q.status === 'wrong' ? '#ff6b6b' : tooltipInfo.q.status === 'correct' ? '#6bffa0' : '#888',
            }} />
            <span style={{ color: tooltipInfo.q.status === 'wrong' ? '#ff7b72' : tooltipInfo.q.status === 'correct' ? '#3fb950' : '#8b949e', fontSize: 10 }}>
              {tooltipInfo.q.status === 'wrong' ? '错误' : tooltipInfo.q.status === 'correct' ? '正确' : '未测试'}
            </span>
            <span style={{ color: '#d29922', fontSize: 10 }}>{'★'.repeat(tooltipInfo.q.difficulty || 0)}</span>
          </div>
          <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{tooltipInfo.q.content}</div>
        </div>
      )}

      {/* Click-to-open question detail panel (centered modal) */}
      {detailQ && (
        <>
          <div style={{ position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.45)', zIndex: 55 }}
            onClick={() => { setDetailQ(null); setAnsweringDetailQ(false); setDetailQAnswerResult(null); setDetailQFailedKPs([]); }} />
          <div style={{
            position: 'absolute', left: '50%', top: '50%', transform: 'translate(-50%, -50%)',
            background: '#161b22', border: '1px solid #30363d', borderRadius: 12,
            padding: 20, width: 420, maxWidth: 'calc(100% - 32px)', maxHeight: '80vh', overflow: 'auto',
            zIndex: 60, boxShadow: '0 8px 32px rgba(0,0,0,0.6)', fontSize: 13, color: '#c8d0e0',
          }} onClick={e => e.stopPropagation()}>
            {/* Header */}
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 16, gap: 8 }}>
              <div style={{ fontSize: 15, color: '#e6edf3', lineHeight: 1.5, fontWeight: 500, flex: 1 }}>{detailQ.content}</div>
              <button style={{ background: 'transparent', border: 'none', color: '#8b949e', cursor: 'pointer', fontSize: 18, flexShrink: 0, lineHeight: 1, padding: 0 }}
                onClick={() => { setDetailQ(null); setAnsweringDetailQ(false); setDetailQAnswerResult(null); setDetailQFailedKPs([]); }}>✕</button>
            </div>
            {/* Stats grid */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px 16px', marginBottom: 14, padding: '12px 14px', background: '#0d1117', borderRadius: 8, border: '1px solid #21262d' }}>
              <div>
                <div style={{ color: '#8b949e', fontSize: 11, marginBottom: 3 }}>状态</div>
                <div style={{ color: detailQ.status === 'wrong' ? '#ff7b72' : detailQ.status === 'correct' ? '#3fb950' : '#8b949e', fontWeight: 600 }}>
                  {detailQ.status === 'wrong' ? '✗ 错误' : detailQ.status === 'correct' ? '✓ 正确' : '未测试'}
                </div>
              </div>
              <div>
                <div style={{ color: '#8b949e', fontSize: 11, marginBottom: 3 }}>难度</div>
                <div style={{ color: '#d29922' }}>{'★'.repeat(detailQ.difficulty || 0)}{'☆'.repeat(5 - (detailQ.difficulty || 0))}</div>
              </div>
              <div>
                <div style={{ color: '#8b949e', fontSize: 11, marginBottom: 3 }}>做题次数</div>
                <div style={{ fontWeight: 600 }}>{detailQ.history?.length || 0} 次</div>
              </div>
              <div>
                <div style={{ color: '#8b949e', fontSize: 11, marginBottom: 3 }}>Leitner 格</div>
                <div style={{ fontWeight: 600 }}>第 {(detailQ.leitnerBox || 0) + 1} 格</div>
              </div>
            </div>
            {/* Associated KPs */}
            <div style={{ marginBottom: 10 }}>
              <div style={{ color: '#8b949e', fontSize: 11, marginBottom: 6 }}>关联知识点</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
                {detailQ.knowledgePointIds?.map(kpId => (
                  <span key={kpId} style={{
                    background: 'rgba(99,140,255,0.12)', border: '1px solid rgba(99,140,255,0.3)',
                    color: '#638cff', fontSize: 11, padding: '3px 10px', borderRadius: 10,
                  }}>{nodesTreeRef.current[kpId]?.name || kpId}</span>
                ))}
              </div>
            </div>
            {/* Failed KPs */}
            {detailQ.failedKnowledgePointIds?.length > 0 && (
              <div style={{ marginBottom: 10 }}>
                <div style={{ color: '#8b949e', fontSize: 11, marginBottom: 6 }}>失败知识点</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
                  {detailQ.failedKnowledgePointIds.map(kpId => (
                    <span key={kpId} style={{
                      background: 'rgba(218,54,51,0.15)', border: '1px solid rgba(218,54,51,0.3)',
                      color: '#ff7b72', fontSize: 11, padding: '3px 10px', borderRadius: 10,
                    }}>{nodesTreeRef.current[kpId]?.name || kpId}</span>
                  ))}
                </div>
              </div>
            )}
            {/* History */}
            {detailQ.history?.length > 0 && (
              <div style={{ marginBottom: 14 }}>
                <div style={{ color: '#8b949e', fontSize: 11, marginBottom: 6 }}>最近做题记录</div>
                <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                  {detailQ.history.slice(-8).map((h, i) => (
                    <span key={i} style={{
                      fontSize: 11, padding: '2px 8px', borderRadius: 8,
                      background: h.result === 'correct' ? 'rgba(63,185,80,0.1)' : 'rgba(255,123,114,0.1)',
                      border: `1px solid ${h.result === 'correct' ? 'rgba(63,185,80,0.3)' : 'rgba(255,123,114,0.3)'}`,
                      color: h.result === 'correct' ? '#3fb950' : '#ff7b72',
                    }}>步骤{h.step}: {h.result === 'correct' ? '✓' : '✗'}</span>
                  ))}
                </div>
              </div>
            )}
            {/* Answer result feedback */}
            {(detailQAnswerResult === 'correct' || detailQAnswerResult === 'wrong') && (
              <div style={{
                padding: '12px 16px', borderRadius: 8, marginBottom: 12, textAlign: 'center',
                background: detailQAnswerResult === 'correct' ? 'rgba(35,134,54,0.15)' : 'rgba(218,54,51,0.15)',
                border: `1px solid ${detailQAnswerResult === 'correct' ? 'rgba(35,134,54,0.4)' : 'rgba(218,54,51,0.4)'}`,
                color: detailQAnswerResult === 'correct' ? '#3fb950' : '#ff7b72',
                fontSize: 15, fontWeight: 600,
              }}>
                {detailQAnswerResult === 'correct' ? '✓ 正确！掌握度已更新' : '✗ 错误。继续努力！'}
              </div>
            )}
            {/* KP selection for wrong answer */}
            {detailQAnswerResult === 'selectKP' && (
              <div style={{ marginBottom: 12, padding: 12, background: '#0d1117', borderRadius: 8, border: '1px solid #30363d' }}>
                <div style={{ color: '#8b949e', marginBottom: 8, fontSize: 12 }}>因为哪些知识点不会？（不选则默认全部）</div>
                {detailQ.knowledgePointIds?.map(kpId => (
                  <label key={kpId} style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '4px 0', cursor: 'pointer', fontSize: 13 }}>
                    <input type="checkbox" checked={detailQFailedKPs.includes(kpId)}
                      onChange={e => setDetailQFailedKPs(prev => e.target.checked ? [...prev, kpId] : prev.filter(x => x !== kpId))} />
                    <span>{nodesTreeRef.current[kpId]?.name || kpId}</span>
                  </label>
                ))}
                <button style={{ marginTop: 8, background: '#da3633', color: '#fff', border: 'none', padding: '6px 16px', borderRadius: 6, cursor: 'pointer', fontSize: 12, fontWeight: 600 }}
                  onClick={() => {
                    const fkp = detailQFailedKPs.length > 0 ? detailQFailedKPs : (detailQ.knowledgePointIds || []);
                    onAnswerQuestion(detailQ.id, false, fkp);
                    setDetailQAnswerResult('wrong'); setDetailQFailedKPs([]);
                  }}>确认</button>
              </div>
            )}
            {/* Inline answer UI */}
            {answeringDetailQ && !detailQAnswerResult && (
              <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
                <button style={{ flex: 1, padding: '9px 12px', background: '#238636', color: '#fff', border: 'none', borderRadius: 6, cursor: 'pointer', fontSize: 13, fontWeight: 600 }}
                  onClick={() => {
                    onAnswerQuestion(detailQ.id, true, []);
                    setDetailQAnswerResult('correct'); setAnsweringDetailQ(false);
                  }}>✓ 做对了</button>
                <button style={{ flex: 1, padding: '9px 12px', background: '#da3633', color: '#fff', border: 'none', borderRadius: 6, cursor: 'pointer', fontSize: 13, fontWeight: 600 }}
                  onClick={() => {
                    if ((detailQ.knowledgePointIds || []).length <= 1) {
                      onAnswerQuestion(detailQ.id, false, detailQ.knowledgePointIds || []);
                      setDetailQAnswerResult('wrong'); setAnsweringDetailQ(false);
                    } else {
                      setDetailQAnswerResult('selectKP'); setAnsweringDetailQ(false);
                    }
                  }}>✗ 做错了</button>
                <button style={{ padding: '9px 12px', background: 'transparent', color: '#8b949e', border: '1px solid #30363d', borderRadius: 6, cursor: 'pointer', fontSize: 13 }}
                  onClick={() => setAnsweringDetailQ(false)}>取消</button>
              </div>
            )}
            {/* Action buttons */}
            {!answeringDetailQ && !detailQAnswerResult && (
              <div style={{ display: 'flex', gap: 8 }}>
                <button style={{ flex: 1, padding: '9px 12px', background: 'rgba(99,140,255,0.12)', color: '#638cff', border: '1px solid rgba(99,140,255,0.3)', borderRadius: 6, cursor: 'pointer', fontSize: 13, fontWeight: 500 }}
                  onClick={() => setAnsweringDetailQ(true)}>▶ 做这道题</button>
                <button style={{ padding: '9px 14px', background: 'rgba(218,54,51,0.1)', color: '#ff7b72', border: '1px solid rgba(218,54,51,0.3)', borderRadius: 6, cursor: 'pointer', fontSize: 13 }}
                  onClick={() => { onDeleteQuestion(detailQ.id); setDetailQ(null); }}>删除</button>
              </div>
            )}
            {detailQAnswerResult && detailQAnswerResult !== 'selectKP' && (
              <div style={{ display: 'flex', justifyContent: 'center', marginTop: 8 }}>
                <button style={{ padding: '7px 24px', background: '#21262d', color: '#c8d0e0', border: '1px solid #30363d', borderRadius: 6, cursor: 'pointer', fontSize: 13 }}
                  onClick={() => { setDetailQAnswerResult(null); setDetailQFailedKPs([]); }}>关闭</button>
              </div>
            )}
          </div>
        </>
      )}

      {contextMenu && (
        <div style={{
          position: 'absolute', left: contextMenu.x, top: contextMenu.y,
          background: '#21262d', border: '1px solid #30363d', borderRadius: 8,
          padding: 4, zIndex: 100, minWidth: 160, boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
        }} onClick={e => e.stopPropagation()}>
          <div style={styles.ctxMenuTitle}>{contextMenu.nodeName}</div>
          <button style={styles.ctxMenuItem} onClick={() => { onNodeClick(contextMenu.nodeId); setContextMenu(null); }}>📋 查看详情</button>
          <button style={styles.ctxMenuItem} onClick={() => { onNodeDblClick(contextMenu.nodeId); setContextMenu(null); }}>📂 展开/折叠</button>
          <button style={styles.ctxMenuItem} onClick={() => {
            setShowQuestionsFor(prev => prev === contextMenu.nodeId ? null : contextMenu.nodeId); setContextMenu(null);
          }}>🔍 {showQuestionsFor === contextMenu.nodeId ? '隐藏题目' : '显示题目'}</button>
          <div style={{ borderTop: '1px solid #30363d', margin: '2px 0' }} />
          <button style={styles.ctxMenuItem} onClick={() => { onAddChild(contextMenu.nodeId); setContextMenu(null); }}>＋ 添加子知识点</button>
          <button style={styles.ctxMenuItem} onClick={() => { onAddQuestion(contextMenu.nodeId); setContextMenu(null); }}>＋ 添加题目</button>
          <div style={{ borderTop: '1px solid #30363d', margin: '2px 0' }} />
          <button style={{ ...styles.ctxMenuItem, color: '#ff7b72' }} onClick={() => { onDeleteNode(contextMenu.nodeId); setContextMenu(null); }}>✕ 删除知识点</button>
        </div>
      )}
    </div>
  );
}

// ============ REVIEW PANEL ============
function ReviewPanel({ question, questions, reviewIdx, total, nodes, allLeafNodes, isLeaf,
  failedKPs, setFailedKPs, reviewResult, setReviewResult, onAnswer, onNext, onExit }) {
  
  if (!question) return (
    <div style={styles.reviewDone}>
      <div style={styles.reviewDoneIcon}>✓</div>
      <div style={styles.reviewDoneText}>本组复习完成！</div>
      <button style={styles.formBtn} onClick={onExit}>返回图谱</button>
    </div>
  );

  return (
    <div style={styles.reviewContainer}>
      <div style={{ width: '100%', maxWidth: 600, marginBottom: 20 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
          <span style={{ fontSize: 13, color: '#8b949e' }}>复习进度 {reviewIdx + 1} / {total}</span>
          <button style={styles.reviewExit} onClick={onExit}>✕ 退出</button>
        </div>
        <div style={{ height: 5, background: '#21262d', borderRadius: 3, overflow: 'hidden' }}>
          <div style={{ height: '100%', borderRadius: 3, background: 'linear-gradient(90deg, #238636, #2ea043)', transition: 'width 0.4s ease', width: `${((reviewIdx + 1) / total) * 100}%` }} />
        </div>
      </div>
      <div style={styles.reviewCard}>
        <div style={styles.reviewQContent}>{question.content}</div>
        <div style={styles.reviewQMeta}>
          <span>难度 {'★'.repeat(question.difficulty)}{'☆'.repeat(5 - question.difficulty)}</span>
          <span>关联：{question.knowledgePointIds.map(k => nodes[k]?.name || '?').join('、')}</span>
        </div>
        {question.status === 'wrong' && (
          <div style={styles.reviewPrevWrong}>上次错误原因：{question.failedKnowledgePointIds.map(k => nodes[k]?.name || '?').join('、')}</div>
        )}
      </div>

      {!reviewResult && (
        <div style={styles.reviewActions}>
          <button style={styles.reviewCorrectBtn} onClick={() => onAnswer(true)}>✓ 做对了</button>
          <button style={styles.reviewWrongBtn} onClick={() => {
            if (question.knowledgePointIds.length === 1) {
              setFailedKPs(question.knowledgePointIds);
              onAnswer(false);
            } else {
              setReviewResult('selectKP');
            }
          }}>✗ 做错了</button>
        </div>
      )}

      {reviewResult === 'selectKP' && (
        <div style={styles.reviewSelectKP}>
          <div style={styles.formLabel}>因为哪些知识点不会？</div>
          {question.knowledgePointIds.map(kpId => (
            <label key={kpId} style={styles.kpOption}>
              <input type="checkbox" checked={failedKPs.includes(kpId)}
                onChange={e => setFailedKPs(prev => e.target.checked ? [...prev, kpId] : prev.filter(x => x !== kpId))} />
              <span style={{ marginLeft: 6 }}>{nodes[kpId]?.name || kpId}</span>
            </label>
          ))}
          <button style={styles.reviewWrongBtn}
            disabled={failedKPs.length === 0}
            onClick={() => onAnswer(false)}>确认</button>
        </div>
      )}

      {(reviewResult === 'correct' || reviewResult === 'wrong') && (
        <div style={styles.reviewFeedback}>
          <div style={reviewResult === 'correct' ? styles.reviewFeedbackCorrect : styles.reviewFeedbackWrong}>
            {reviewResult === 'correct' ? '✓ 正确！' : '✗ 错误'}
          </div>
          <button style={styles.formBtn} onClick={onNext}>
            {reviewIdx + 1 < total ? '下一题 →' : '完成复习'}
          </button>
        </div>
      )}
    </div>
  );
}

// ============ STYLES ============
const styles = {
  container: { display: 'flex', flexDirection: 'column', height: '100vh', background: '#0d1117', color: '#c8d0e0', fontFamily: "'SF Pro Text', system-ui, -apple-system, sans-serif" },
  loading: { display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', fontSize: 18, color: '#888', background: '#0d1117' },
  topBar: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 20px', background: '#161b22', borderBottom: '1px solid #21262d', zIndex: 10 },
  logo: { fontSize: 16, fontWeight: 600, color: '#e6edf3', display: 'flex', alignItems: 'center', gap: 8 },
  logoIcon: { fontSize: 20, color: '#638cff' },
  topActions: { display: 'flex', alignItems: 'center', gap: 12 },
  statBadge: { fontSize: 12, color: '#8b949e', background: '#21262d', padding: '4px 10px', borderRadius: 12 },
  reviewBtn: { background: '#238636', color: '#fff', border: 'none', padding: '6px 16px', borderRadius: 6, fontSize: 13, fontWeight: 600, cursor: 'pointer' },
  sampleBtn: { background: '#1f6feb', color: '#fff', border: 'none', padding: '6px 14px', borderRadius: 6, fontSize: 12, fontWeight: 600, cursor: 'pointer' },
  resetBtn: { background: 'transparent', color: '#8b949e', border: '1px solid #30363d', padding: '5px 12px', borderRadius: 6, fontSize: 12, cursor: 'pointer' },
  mainLayout: { display: 'flex', flex: 1, overflow: 'hidden' },
  sidePanel: { width: 300, minWidth: 300, background: '#161b22', borderRight: '1px solid #21262d', display: 'flex', flexDirection: 'column', overflow: 'hidden' },
  sideTabs: { display: 'flex', borderBottom: '1px solid #21262d' },
  sideTab: { flex: 1, padding: '8px 4px', background: 'transparent', border: 'none', color: '#8b949e', fontSize: 12, cursor: 'pointer', borderBottom: '2px solid transparent' },
  sideTabActive: { flex: 1, padding: '8px 4px', background: 'transparent', border: 'none', color: '#638cff', fontSize: 12, cursor: 'pointer', borderBottom: '2px solid #638cff', fontWeight: 600 },
  sidePanelContent: { flex: 1, overflow: 'auto' },
  graphArea: { flex: 1, position: 'relative', overflow: 'hidden' },
  emptyState: { display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: 12, color: '#484f58' },
  emptyIcon: { fontSize: 48, opacity: 0.3 },
  emptyTitle: { fontSize: 18, fontWeight: 500 },
  emptyHint: { fontSize: 13, color: '#6e7681' },
  // Tree
  treeContainer: { padding: '4px 0' },
  treeEmpty: { padding: 20, textAlign: 'center', color: '#484f58', fontSize: 13 },
  treeItem: { display: 'flex', alignItems: 'center', gap: 6, padding: '5px 8px', cursor: 'pointer', fontSize: 13, transition: 'background 0.15s', borderRadius: 0 },
  treeToggle: { fontSize: 11, color: '#8b949e', width: 14, textAlign: 'center', flexShrink: 0 },
  treeLeafDot: { fontSize: 6, color: '#484f58', width: 14, textAlign: 'center', flexShrink: 0 },
  treeMasteryDot: { width: 8, height: 8, borderRadius: '50%', flexShrink: 0 },
  treeNodeName: { flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
  treeErrorBadge: { background: '#da3633', color: '#fff', fontSize: 10, padding: '1px 5px', borderRadius: 8, fontWeight: 600 },
  // Forms
  formPanel: { padding: 16, display: 'flex', flexDirection: 'column', gap: 10 },
  formLabel: { fontSize: 12, color: '#8b949e', fontWeight: 500, marginTop: 4 },
  formInput: { background: '#0d1117', border: '1px solid #30363d', borderRadius: 6, padding: '8px 10px', color: '#e6edf3', fontSize: 13, outline: 'none' },
  formTextarea: { background: '#0d1117', border: '1px solid #30363d', borderRadius: 6, padding: '8px 10px', color: '#e6edf3', fontSize: 13, outline: 'none', resize: 'vertical', fontFamily: 'inherit' },
  formSelect: { background: '#0d1117', border: '1px solid #30363d', borderRadius: 6, padding: '8px 10px', color: '#e6edf3', fontSize: 13, outline: 'none' },
  formBtn: { background: '#238636', color: '#fff', border: 'none', padding: '8px 16px', borderRadius: 6, fontSize: 13, fontWeight: 600, cursor: 'pointer', marginTop: 4 },
  kpSelector: { maxHeight: 150, overflow: 'auto', background: '#0d1117', border: '1px solid #30363d', borderRadius: 6, padding: 8 },
  kpOption: { display: 'flex', alignItems: 'center', padding: '3px 0', cursor: 'pointer', fontSize: 13 },
  difficultyRow: { display: 'flex', gap: 6 },
  diffBtn: { width: 32, height: 32, borderRadius: 6, border: '1px solid #30363d', background: '#0d1117', color: '#8b949e', cursor: 'pointer', fontSize: 13 },
  diffBtnActive: { width: 32, height: 32, borderRadius: 6, border: '1px solid #638cff', background: 'rgba(99,140,255,0.15)', color: '#638cff', cursor: 'pointer', fontSize: 13, fontWeight: 700 },
  statusRow: { display: 'flex', gap: 6 },
  statusBtn: { flex: 1, padding: '6px 8px', borderRadius: 6, border: '1px solid #30363d', background: '#0d1117', color: '#8b949e', cursor: 'pointer', fontSize: 12 },
  statusBtnCorrect: { flex: 1, padding: '6px 8px', borderRadius: 6, border: '1px solid #238636', background: 'rgba(35,134,54,0.15)', color: '#3fb950', cursor: 'pointer', fontSize: 12, fontWeight: 600 },
  statusBtnWrong: { flex: 1, padding: '6px 8px', borderRadius: 6, border: '1px solid #da3633', background: 'rgba(218,54,51,0.15)', color: '#ff7b72', cursor: 'pointer', fontSize: 12, fontWeight: 600 },
  statusBtnActive: { flex: 1, padding: '6px 8px', borderRadius: 6, border: '1px solid #638cff', background: 'rgba(99,140,255,0.15)', color: '#638cff', cursor: 'pointer', fontSize: 12, fontWeight: 600 },
  // Detail panel
  detailPanel: { borderTop: '1px solid #21262d', padding: 16, maxHeight: 300, overflow: 'auto' },
  detailTitle: { fontSize: 15, fontWeight: 600, color: '#e6edf3', marginBottom: 2 },
  detailPath: { fontSize: 11, color: '#484f58', marginBottom: 12 },
  detailStats: { display: 'flex', flexDirection: 'column', gap: 8 },
  detailStatRow: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 12, gap: 8 },
  masteryBar: { flex: 1, height: 6, background: '#21262d', borderRadius: 3, overflow: 'hidden' },
  masteryBarFill: { height: '100%', borderRadius: 3, transition: 'width 0.3s' },
  detailQList: { marginTop: 12 },
  detailQListTitle: { fontSize: 12, color: '#8b949e', marginBottom: 6, fontWeight: 500 },
  detailQItem: { display: 'flex', alignItems: 'center', gap: 6, padding: '4px 0', fontSize: 12, borderBottom: '1px solid #21262d' },
  qStatusDot: { width: 6, height: 6, borderRadius: '50%', flexShrink: 0 },
  qContent: { flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
  qDiff: { color: '#d29922', fontSize: 10, flexShrink: 0 },
  deleteNodeBtn: { background: 'transparent', border: '1px solid #30363d', color: '#ff7b72', width: 24, height: 24, borderRadius: 4, cursor: 'pointer', fontSize: 14, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 },
  deleteQBtn: { background: 'transparent', border: 'none', color: '#484f58', cursor: 'pointer', fontSize: 16, padding: '0 2px', flexShrink: 0, lineHeight: 1 },
  qDetail: { padding: '6px 8px 8px 22px', background: 'rgba(99,140,255,0.05)', borderBottom: '1px solid #21262d', fontSize: 11, color: '#8b949e', display: 'flex', flexDirection: 'column', gap: 3 },
  qDetailRow: { display: 'flex', gap: 4 },
  ctxMenuTitle: { padding: '6px 12px', fontSize: 12, color: '#8b949e', borderBottom: '1px solid #30363d', fontWeight: 500 },
  ctxMenuItem: { display: 'block', width: '100%', padding: '8px 12px', background: 'transparent', border: 'none', color: '#c8d0e0', fontSize: 13, cursor: 'pointer', textAlign: 'left', borderRadius: 4 },
  kpTag: { display: 'inline-flex', alignItems: 'center', gap: 4, background: 'rgba(99,140,255,0.15)', border: '1px solid rgba(99,140,255,0.3)', color: '#638cff', fontSize: 11, padding: '2px 8px', borderRadius: 10 },
  kpTagX: { cursor: 'pointer', opacity: 0.6, fontSize: 13, marginLeft: 2 },
  // Review
  reviewContainer: { display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', padding: 40, background: '#0d1117' },
  reviewHeader: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%', maxWidth: 600, marginBottom: 24 },
  reviewExit: { background: 'transparent', border: '1px solid #30363d', color: '#8b949e', padding: '4px 12px', borderRadius: 6, cursor: 'pointer', fontSize: 13 },
  reviewCard: { background: '#161b22', border: '1px solid #30363d', borderRadius: 12, padding: 24, width: '100%', maxWidth: 600, marginBottom: 24 },
  reviewQContent: { fontSize: 16, color: '#e6edf3', lineHeight: 1.6, marginBottom: 16 },
  reviewQMeta: { display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#8b949e' },
  reviewPrevWrong: { marginTop: 12, padding: '8px 12px', background: 'rgba(218,54,51,0.1)', borderRadius: 6, fontSize: 12, color: '#ff7b72' },
  reviewActions: { display: 'flex', gap: 16, width: '100%', maxWidth: 600 },
  reviewCorrectBtn: { flex: 1, padding: '12px 20px', borderRadius: 8, border: 'none', background: '#238636', color: '#fff', fontSize: 15, fontWeight: 600, cursor: 'pointer' },
  reviewWrongBtn: { flex: 1, padding: '12px 20px', borderRadius: 8, border: 'none', background: '#da3633', color: '#fff', fontSize: 15, fontWeight: 600, cursor: 'pointer' },
  reviewSelectKP: { width: '100%', maxWidth: 600, background: '#161b22', border: '1px solid #30363d', borderRadius: 12, padding: 20, display: 'flex', flexDirection: 'column', gap: 8 },
  reviewFeedback: { display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16, width: '100%', maxWidth: 600 },
  reviewFeedbackCorrect: { fontSize: 24, color: '#3fb950', fontWeight: 700 },
  reviewFeedbackWrong: { fontSize: 24, color: '#ff7b72', fontWeight: 700 },
  reviewDone: { display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: 16 },
  reviewDoneIcon: { fontSize: 48, color: '#3fb950' },
  reviewDoneText: { fontSize: 20, color: '#e6edf3', fontWeight: 600 },
};

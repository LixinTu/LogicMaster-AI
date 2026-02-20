import { getNextDemoQuestion, DEMO_THETA_RESPONSE, DEMO_REVIEW_SCHEDULE, DEMO_TUTOR_START, DEMO_TUTOR_CONTINUE } from './demo-data';
import { API_BASE_URL } from './config';

const BASE_URL = API_BASE_URL;

let _isDemoMode = false;
export function isDemoMode() { return _isDemoMode; }

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const token = localStorage.getItem('auth_token');
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  const res = await fetch(`${BASE_URL}${path}`, {
    headers,
    ...options,
  });
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  // API call succeeded — clear demo mode
  _isDemoMode = false;
  return res.json();
}

// ─── Types ──────────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  env: string;
  db_status: string;
  qdrant_status: string;
}

export interface NextQuestionResponse {
  question_id: string;
  question_type: string;
  difficulty: string;
  stimulus: string;
  question: string;
  choices: string[];
  skills: string[];
  elo_difficulty: number;
  correct_answer: string; // Letter A-E
}

export interface ThetaResponse {
  new_theta: number;
  gmat_score: number;
}

export interface TutorStartResponse {
  conversation_id: string;
  first_hint: string;
  logic_gap: string;
  error_type: string;
  hint_count: number;
  student_understanding: string;
  current_state: string;
  variant: string;
}

export interface TutorContinueResponse {
  reply: string;
  hint_count: number;
  blooms_level: number;
  blooms_name: string;
  student_understanding: string;
  should_continue: boolean;
  current_state: string;
}

export interface ReviewItem {
  question_id: string;
  question_type?: string;
  difficulty?: string;
  stimulus_preview?: string;
  recall_probability: number;
  half_life: number;
  elapsed_days: number;
  skills?: string[];
}

export interface ReviewScheduleResponse {
  user_id: string;
  threshold: number;
  due_count: number;
  reviews: ReviewItem[];
}

export interface DashboardSummary {
  today_goal: number;
  today_completed: number;
  streak_days: number;
  current_theta: number;
  gmat_score: number;
  accuracy_pct: number;
  total_questions: number;
  weak_skills: { skill_name: string; error_rate: number; mastery: number }[];
  reviews_due: number;
  last_practiced: string;
  last_7_days: boolean[];
}

export interface BookmarkItem {
  question_id: string;
  question_type: string;
  difficulty: string;
  stimulus_preview: string;
  skills: string[];
  bookmark_type: 'wrong' | 'favorite';
  created_at: string;
}

export interface BookmarkStats {
  total_wrong: number;
  by_skill: { skill_name: string; count: number }[];
  by_type: { question_type: string; count: number }[];
}

export interface GoalProgress {
  target_gmat_score: number;
  daily_question_goal: number;
  current_gmat_score: number;
  score_gap: number;
  estimated_questions_remaining: number;
  today_completed: number;
  on_track: boolean;
  last_7_days: boolean[];
}

export interface UserStats {
  total_questions: number;
  total_correct: number;
  accuracy_pct: number;
  best_streak: number;
  current_gmat_score: number;
  current_theta: number;
  member_since: string;
  favorite_question_type: string;
}

export interface AnalyticsData {
  wrong_by_type: { name: string; value: number; color: string }[];
  wrong_by_skill: { skill: string; count: number }[];
  skill_mastery: { skill: string; value: number }[];
  answer_history: { question_id: string; is_correct: boolean; theta_at_time: number; timestamp?: string }[];
}

// ─── Demo fallbacks ──────────────────────────────────────────────

const DEMO_DASHBOARD: DashboardSummary = {
  today_goal: 5,
  today_completed: 0,
  streak_days: 0,
  current_theta: 0,
  gmat_score: 25,
  accuracy_pct: 0,
  total_questions: 0,
  weak_skills: [],
  reviews_due: 0,
  last_practiced: '',
  last_7_days: [false, false, false, false, false, false, false],
};

const DEMO_GOAL_PROGRESS: GoalProgress = {
  target_gmat_score: 40,
  daily_question_goal: 5,
  current_gmat_score: 25,
  score_gap: 15,
  estimated_questions_remaining: 0,
  today_completed: 0,
  on_track: false,
  last_7_days: [false, false, false, false, false, false, false],
};

// ─── API ──────────────────────────────────────────────────────

export const api = {
  health: () => request<HealthResponse>('/health'),

  // ── Dashboard ───────────────────────────────────────────────
  dashboardSummary: async (userId: string) => {
    try {
      return await request<DashboardSummary>(`/api/dashboard/summary?user_id=${userId}`);
    } catch {
      _isDemoMode = true;
      return DEMO_DASHBOARD;
    }
  },

  // ── Questions ───────────────────────────────────────────────
  getQuestionById: async (questionId: string) => {
    try {
      return await request<NextQuestionResponse>(`/api/questions/${questionId}`);
    } catch {
      _isDemoMode = true;
      const { DEMO_QUESTIONS } = await import('./demo-data');
      const found = DEMO_QUESTIONS.find(q => q.question_id === questionId);
      return found || getNextDemoQuestion();
    }
  },

  nextQuestion: async (userTheta: number, currentQId: string, questionsLog: any[], strategy = 'bandit') => {
    try {
      return await request<NextQuestionResponse>('/api/questions/next', {
        method: 'POST',
        body: JSON.stringify({ user_theta: userTheta, current_q_id: currentQId, questions_log: questionsLog, strategy }),
      });
    } catch {
      _isDemoMode = true;
      return getNextDemoQuestion();
    }
  },

  updateTheta: async (currentTheta: number, questionDifficulty: number, isCorrect: boolean) => {
    try {
      return await request<ThetaResponse>('/api/theta/update', {
        method: 'POST',
        body: JSON.stringify({ current_theta: currentTheta, question_difficulty: questionDifficulty, is_correct: isCorrect }),
      });
    } catch {
      _isDemoMode = true;
      const delta = isCorrect ? 0.08 : -0.05;
      return { new_theta: currentTheta + delta, gmat_score: Math.round(550 + (currentTheta + delta) * 100) };
    }
  },

  // ── Tutoring ────────────────────────────────────────────────
  startRemediation: async (questionId: string, question: any, userChoice: string, correctChoice: string, userId: string) => {
    try {
      return await request<TutorStartResponse>('/api/tutor/start-remediation', {
        method: 'POST',
        body: JSON.stringify({ question_id: questionId, question, user_choice: userChoice, correct_choice: correctChoice, user_id: userId }),
      });
    } catch {
      _isDemoMode = true;
      return DEMO_TUTOR_START;
    }
  },

  continueTutor: async (conversationId: string, userMessage: string, questionId: string) => {
    try {
      return await request<TutorContinueResponse>('/api/tutor/continue', {
        method: 'POST',
        body: JSON.stringify({ conversation_id: conversationId, student_message: userMessage, question_id: questionId }),
      });
    } catch {
      _isDemoMode = true;
      return DEMO_TUTOR_CONTINUE;
    }
  },

  concludeTutor: async (conversationId: string, questionId: string) => {
    try {
      return await request<any>('/api/tutor/conclude', {
        method: 'POST',
        body: JSON.stringify({ conversation_id: conversationId, question_id: questionId }),
      });
    } catch {
      _isDemoMode = true;
      return { status: 'concluded' };
    }
  },

  generateExplanation: async (questionId: string, question: any, userChoice: string, isCorrect: boolean) => {
    try {
      return await request<any>('/api/explanations/generate-with-rag', {
        method: 'POST',
        body: JSON.stringify({ question_id: questionId, question, user_choice: userChoice, is_correct: isCorrect }),
      });
    } catch {
      _isDemoMode = true;
      return 'The correct answer addresses the key logical flaw in the argument.';
    }
  },

  // ── Analytics ───────────────────────────────────────────────
  abTestResults: (experiment = 'tutor_strategy') =>
    request<any>(`/api/analytics/ab-test-results?experiment=${experiment}`),

  ragPerformance: () => request<any>('/api/analytics/rag-performance'),

  analyticsData: async (userId: string) => {
    try {
      return await request<AnalyticsData>(`/api/analytics/summary?user_id=${userId}`);
    } catch {
      _isDemoMode = true;
      return null;
    }
  },

  // ── Review ──────────────────────────────────────────────────
  reviewSchedule: async (userId: string, threshold = 0.5) => {
    try {
      return await request<ReviewScheduleResponse>(`/api/questions/review-schedule?user_id=${userId}&threshold=${threshold}`);
    } catch {
      _isDemoMode = true;
      return DEMO_REVIEW_SCHEDULE;
    }
  },

  banditUpdate: async (questionId: string, isCorrect: boolean, skills: string[], thetaAtTime: number, userId: string) => {
    try {
      return await request<any>('/api/questions/bandit-update', {
        method: 'POST',
        body: JSON.stringify({ question_id: questionId, is_correct: isCorrect, skills, theta_at_time: thetaAtTime, user_id: userId }),
      });
    } catch {
      _isDemoMode = true;
      return { status: 'ok' };
    }
  },

  // ── Bookmarks / Wrong Book ──────────────────────────────────
  bookmarksList: async (userId: string) => {
    try {
      return await request<BookmarkItem[]>(`/api/bookmarks/list?user_id=${userId}`);
    } catch {
      _isDemoMode = true;
      return [];
    }
  },

  bookmarksStats: async (userId: string) => {
    try {
      return await request<BookmarkStats>(`/api/bookmarks/wrong-stats?user_id=${userId}`);
    } catch {
      _isDemoMode = true;
      return { total_wrong: 0, by_skill: [], by_type: [] };
    }
  },

  bookmarkRemove: async (userId: string, questionId: string, bookmarkType: 'wrong' | 'favorite' = 'wrong') => {
    try {
      return await request<any>(`/api/bookmarks/remove`, {
        method: 'DELETE',
        body: JSON.stringify({ user_id: userId, question_id: questionId, bookmark_type: bookmarkType }),
      });
    } catch {
      return { status: 'ok' };
    }
  },

  bookmarkAdd: async (userId: string, questionId: string, bookmarkType: 'wrong' | 'favorite') => {
    try {
      return await request<any>(`/api/bookmarks/add`, {
        method: 'POST',
        body: JSON.stringify({ user_id: userId, question_id: questionId, bookmark_type: bookmarkType }),
      });
    } catch {
      return { status: 'ok' };
    }
  },

  // ── Goals ───────────────────────────────────────────────────
  goalProgress: async (userId: string) => {
    try {
      return await request<GoalProgress>(`/api/goals/progress?user_id=${userId}`);
    } catch {
      _isDemoMode = true;
      return DEMO_GOAL_PROGRESS;
    }
  },

  setGoal: async (userId: string, targetGmatScore: number, dailyGoal: number) => {
    try {
      return await request<any>('/api/goals/set', {
        method: 'POST',
        body: JSON.stringify({ user_id: userId, target_gmat_score: targetGmatScore, daily_question_goal: dailyGoal }),
      });
    } catch {
      _isDemoMode = true;
      return { status: 'ok' };
    }
  },

  // ── Profile / Auth ──────────────────────────────────────────
  authStats: async () => {
    try {
      return await request<UserStats>('/api/auth/stats');
    } catch {
      _isDemoMode = true;
      return null;
    }
  },

  updateProfile: async (displayName: string) => {
    return await request<any>('/api/auth/profile', {
      method: 'PUT',
      body: JSON.stringify({ display_name: displayName }),
    });
  },

  changePassword: async (currentPassword: string, newPassword: string) => {
    return await request<any>('/api/auth/change-password', {
      method: 'PUT',
      body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
    });
  },

  deleteAccount: async () => {
    return await request<any>('/api/auth/account', {
      method: 'DELETE',
    });
  },
};

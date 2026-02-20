import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface Question {
  question_id: string;
  question_type: string;
  difficulty: string;
  stimulus: string;
  question: string;
  choices: string[];
  skills: string[];
  elo_difficulty: number;
}

export interface QuestionLog {
  question_id: string;
  is_correct: boolean;
  theta_at_time: number;
}

export interface TutorMessage {
  role: 'system' | 'user';
  content: string;
}

interface AppState {
  // User session
  theta: number;
  gmatScore: number;
  streak: number;
  sessionQuestions: number;
  accuracy: number;
  totalCorrect: number;
  totalAnswered: number;
  questionsLog: QuestionLog[];

  // Current question
  currentQuestion: Question | null;
  selectedChoice: string | null;
  practiceState: 'loading' | 'answering' | 'correct' | 'wrong' | 'tutoring' | 'retrying';

  // Tutoring
  conversationId: string | null;
  tutorMessages: TutorMessage[];
  bloomsLevel: number;
  bloomsName: string;
  hintCount: number;
  tutorLoading: boolean;

  // API keys
  deepseekApiKey: string;
  openaiApiKey: string;

  // Settings
  animationIntensity: 'off' | 'subtle' | 'full';

  // Actions
  setTheta: (theta: number) => void;
  setGmatScore: (score: number) => void;
  setStreak: (streak: number) => void;
  incrementSession: () => void;
  recordAnswer: (correct: boolean) => void;
  addQuestionLog: (log: QuestionLog) => void;
  setCurrentQuestion: (q: Question | null) => void;
  setSelectedChoice: (choice: string | null) => void;
  setPracticeState: (state: AppState['practiceState']) => void;
  setConversationId: (id: string | null) => void;
  addTutorMessage: (msg: TutorMessage) => void;
  clearTutorMessages: () => void;
  setBloomsLevel: (level: number) => void;
  setBloomsName: (name: string) => void;
  setHintCount: (count: number) => void;
  setTutorLoading: (loading: boolean) => void;
  setDeepseekApiKey: (key: string) => void;
  setOpenaiApiKey: (key: string) => void;
  setAnimationIntensity: (intensity: AppState['animationIntensity']) => void;
  resetPracticeState: () => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      theta: 0.0,
      gmatScore: 25,
      streak: 0,
      sessionQuestions: 0,
      accuracy: 0,
      totalCorrect: 0,
      totalAnswered: 0,
      questionsLog: [],
      currentQuestion: null,
      selectedChoice: null,
      practiceState: 'loading',
      conversationId: null,
      tutorMessages: [],
      bloomsLevel: 1,
      bloomsName: 'Remember',
      hintCount: 0,
      tutorLoading: false,
      deepseekApiKey: '',
      openaiApiKey: '',
      animationIntensity: 'full',

      setTheta: (theta) => set({ theta }),
      setGmatScore: (gmatScore) => set({ gmatScore }),
      setStreak: (streak) => set({ streak }),
      incrementSession: () => set((s) => ({ sessionQuestions: s.sessionQuestions + 1 })),
      recordAnswer: (correct) =>
        set((s) => {
          const totalCorrect = s.totalCorrect + (correct ? 1 : 0);
          const totalAnswered = s.totalAnswered + 1;
          return {
            totalCorrect,
            totalAnswered,
            accuracy: Math.round((totalCorrect / totalAnswered) * 100),
            streak: correct ? s.streak + 1 : 0,
          };
        }),
      addQuestionLog: (log) => set((s) => ({ questionsLog: [...s.questionsLog, log] })),
      setCurrentQuestion: (currentQuestion) => set({ currentQuestion }),
      setSelectedChoice: (selectedChoice) => set({ selectedChoice }),
      setPracticeState: (practiceState) => set({ practiceState }),
      setConversationId: (conversationId) => set({ conversationId }),
      addTutorMessage: (msg) => set((s) => ({ tutorMessages: [...s.tutorMessages, msg] })),
      clearTutorMessages: () => set({ tutorMessages: [] }),
      setBloomsLevel: (bloomsLevel) => set({ bloomsLevel }),
      setBloomsName: (bloomsName) => set({ bloomsName }),
      setHintCount: (hintCount) => set({ hintCount }),
      setTutorLoading: (tutorLoading) => set({ tutorLoading }),
      setDeepseekApiKey: (deepseekApiKey) => set({ deepseekApiKey }),
      setOpenaiApiKey: (openaiApiKey) => set({ openaiApiKey }),
      setAnimationIntensity: (animationIntensity) => set({ animationIntensity }),
      resetPracticeState: () =>
        set({
          currentQuestion: null,
          selectedChoice: null,
          practiceState: 'loading',
          conversationId: null,
          tutorMessages: [],
          bloomsLevel: 1,
          bloomsName: 'Remember',
          hintCount: 0,
          tutorLoading: false,
        }),
    }),
    {
      name: 'glitchmind-storage',
      partialize: (state) => ({
        theta: state.theta,
        gmatScore: state.gmatScore,
        streak: state.streak,
        totalCorrect: state.totalCorrect,
        totalAnswered: state.totalAnswered,
        accuracy: state.accuracy,
        questionsLog: state.questionsLog,
        deepseekApiKey: state.deepseekApiKey,
        openaiApiKey: state.openaiApiKey,
        animationIntensity: state.animationIntensity,
      }),
    }
  )
);

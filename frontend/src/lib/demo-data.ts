import type { NextQuestionResponse, ThetaResponse, ReviewScheduleResponse, TutorStartResponse, TutorContinueResponse } from './api';

export const DEMO_QUESTIONS: NextQuestionResponse[] = [
  {
    question_id: 'demo001',
    question_type: 'Weaken',
    difficulty: 'Medium',
    stimulus: 'A pharmaceutical company recently conducted a study showing that patients who took their new drug, Cardiflex, experienced a 30% reduction in heart attacks compared to those who took a placebo. The company concluded that Cardiflex is effective at preventing heart attacks and should be prescribed widely to at-risk patients.',
    question: 'Which of the following, if true, most seriously weakens the company\'s conclusion?',
    choices: [
      'A. The study was funded entirely by the pharmaceutical company that manufactures Cardiflex.',
      'B. Patients in the study who took Cardiflex also adopted healthier diets and exercise routines during the trial period.',
      'C. Cardiflex has fewer side effects than most other heart medications currently on the market.',
      'D. The placebo group had a slightly higher average age than the Cardiflex group.',
      'E. Other pharmaceutical companies are developing similar drugs that may compete with Cardiflex.',
    ],
    skills: ['Critical Reasoning', 'Causal Logic'],
    elo_difficulty: 0.45,
    correct_answer: 'B',
  },
  {
    question_id: 'demo002',
    question_type: 'Strengthen',
    difficulty: 'Hard',
    stimulus: 'City officials argue that installing speed cameras on Highway 9 will reduce the number of traffic accidents. They point to a neighboring city where speed cameras were installed last year, after which accidents declined by 20%.',
    question: 'Which of the following, if true, most strengthens the officials\' argument?',
    choices: [
      'A. Highway 9 has a higher speed limit than the road in the neighboring city.',
      'B. The neighboring city did not implement any other traffic safety measures during the same period.',
      'C. Speed cameras are expensive to install and maintain.',
      'D. Traffic volume on Highway 9 has been increasing steadily over the past five years.',
      'E. Some drivers may slow down only in the immediate vicinity of the cameras.',
    ],
    skills: ['Critical Reasoning', 'Evidence Evaluation'],
    elo_difficulty: 0.65,
    correct_answer: 'B',
  },
];

let demoIndex = 0;

export function getNextDemoQuestion(): NextQuestionResponse {
  const q = DEMO_QUESTIONS[demoIndex % DEMO_QUESTIONS.length];
  demoIndex++;
  return q;
}

export const DEMO_THETA_RESPONSE: ThetaResponse = {
  new_theta: 0.15,
  gmat_score: 550,
};

export const DEMO_REVIEW_SCHEDULE: ReviewScheduleResponse = {
  user_id: 'default',
  threshold: 0.5,
  due_count: 4,
  reviews: [
    { question_id: 'demo001', question_type: 'Weaken', difficulty: 'hard', stimulus_preview: 'A study found that employees who work from home report higher job satisfaction...', recall_probability: 0.25, half_life: 1.5, elapsed_days: 4.2, skills: ['Causal Reasoning'] },
    { question_id: 'demo002', question_type: 'Assumption', difficulty: 'medium', stimulus_preview: 'The city council noted that after installing new traffic cameras at major intersections...', recall_probability: 0.42, half_life: 2.0, elapsed_days: 3.1, skills: ['Assumption Identification'] },
    { question_id: 'demo003', question_type: 'Strengthen', difficulty: 'easy', stimulus_preview: 'Researchers found that students who participated in daily meditation sessions...', recall_probability: 0.68, half_life: 5.0, elapsed_days: 2.5, skills: ['Evidence Evaluation'] },
    { question_id: 'demo004', question_type: 'Inference', difficulty: 'hard', stimulus_preview: 'A pharmaceutical company reported that its new drug reduced symptoms in 70% of patients...', recall_probability: 0.15, half_life: 0.8, elapsed_days: 5.0, skills: ['Logical Structure'] },
  ],
};

export const DEMO_TUTOR_START: TutorStartResponse = {
  conversation_id: 'demo-conv-001',
  first_hint: 'Think about what other factors might explain the observed result besides the drug itself.',
  logic_gap: 'It looks like you may have confused correlation with causation here.',
  error_type: 'causal_confusion',
  hint_count: 1,
  student_understanding: 'confused',
  current_state: 'hinting',
  variant: 'socratic_standard',
};

export const DEMO_TUTOR_CONTINUE: TutorContinueResponse = {
  reply: 'Good thinking! Now consider: if the patients changed their behavior during the trial, can we still attribute the improvement solely to the drug?',
  hint_count: 2,
  blooms_level: 3,
  blooms_name: 'Apply',
  student_understanding: 'partial',
  should_continue: true,
  current_state: 'hinting',
};

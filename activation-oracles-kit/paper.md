https://www.alphaxiv.org/abs/2601.10825?chatId=019bd802-6666-7bd4-8eac-1797dc80bd39

[brainstorm](https://www.notion.so/brainstorm-2eed9897ec048025986cc798278043b5?pvs=21)

Based on the citation patterns in recent literature (specifically "Generative AI collective behavior needs an interactionist paradigm" and related surveys on LLM-based agent societies), here are the papers you are looking for.

These papers collectively focus on **multi-agent systems, social simulation, and emergent behavior in Large Language Model (LLM) populations**.

### **1. Mukobi et al., 2023**

- **Title:** *Welfare Diplomacy: Benchmarking Language Model Cooperation*
- **Authors:** Gabriel Mukobi, Hannah Erlebach, Niklas Lauffer, Lewis Hammond, Alan Chan, Jesse Clifton.
- **Link:** [arXiv:2310.08901](https://arxiv.org/abs/2310.08901)

### **2. Abdelnabi et al., 2023**

- **Title:** *LLM-Deliberation: Evaluating LLMs with Interactive Multi-Agent Negotiation Games* (also cited as *Cooperation, Competition, and Maliciousness* in later versions)
- **Authors:** Sahar Abdelnabi, Amr Gomaa, Sarath Sivaprasad, Lea Schönherr, Mario Fritz.
- **Link:** [arXiv:2309.17234](https://arxiv.org/abs/2309.17234)

### **3. Törnberg et al., 2023**

- **Title:** *Simulating social media using large language models to evaluate alternative news feed algorithms*
- **Authors:** Petter Törnberg, Darya Valeeva, Justus Uitermark, Christopher Bail.
- **Link:** [arXiv:2310.05984](https://arxiv.org/abs/2310.05984)

### **4. Vezhnevets et al., 2023**

- **Title:** *Generative agent-based modeling with actions grounded in physical, social, or digital space using Concordia*
- **Authors:** Alexander Sasha Vezhnevets, John P. Agapiou, Avia Aharon, Ron Ziv, Jayd Matyas, et al.
- **Link:** [arXiv:2312.03664](https://arxiv.org/abs/2312.03664)

### **5. Ren et al., 2024**

- **Title:** *Emergence of Social Norms in Large Language Model-based Agent Societies*
- **Authors:** Siyue Ren, Zhaohan Cui, Rui Song, Zhen Wang, Shuyue Hu.
- **Link:** [arXiv:2403.08251](https://arxiv.org/abs/2403.08251)

### **6. Becker, 2024**

- **Title:** *Multi-Agent Large Language Models for Conversational Task-Solving*
- **Author:** Jonas Becker.
- **Link:** [arXiv:2410.22932](https://arxiv.org/abs/2410.22932)

### **7. Sprigler et al., 2024**

- **Title:** *Synergistic Simulations: Multi-Agent Problem Solving with Large Language Models*
- **Authors:** Asher Sprigler, Alexander Drobek, Keagan Weinstock, Wendpanga Tapsoba, Gavin Childress, Andy Dao, Lucas Gral.
- **Link:** [arXiv:2409.13753](https://arxiv.org/abs/2409.13753) (Also appeared in MICS 2024)

### **8. Campedelli et al., 2025**

- **Title:** *I Want to Break Free! Persuasion and Anti-Social Behavior of LLMs in Multi-Agent Settings with Social Hierarchy*
- **Authors:** Gian Maria Campedelli, Nicolò Penzo, Massimo Stefan, Roberto Dessì, Marco Guerini, Bruno Lepri, Jacopo Staiano.
- **Link:** [arXiv:2410.07109](https://arxiv.org/abs/2410.07109) (Often cited as 2025 in forthcoming/accepted lists)

### **9. Liu et al., 2024**

- **Title:** *The Stepwise Deception: Simulating the Evolution from True News to Fake News with LLM Agents*
- **Authors:** Yuhan Liu, Zirui Song, Juntian Zhang, Xiaoqing Zhang, Xiuying Chen, Rui Yan.
- **Link:** [arXiv:2410.19064](https://arxiv.org/abs/2410.19064)

### **10. Lai et al., 2024**

- **Title:** *Evolving AI Collectives to Enhance Human Diversity and Enable Self-Regulation*
- **Authors:** Shiyang Lai, Yujin Potter, Junsol Kim, Richard Zhuang, Dawn Song, James Evans.
- **Link:** [arXiv:2402.05934](https://arxiv.org/abs/2402.05934)

### **11. Ashery et al., 2025**

- **Title:** *Emergent social conventions and collective bias in LLM populations*
- **Authors:** Ariel Flint Ashery, Luca Maria Aiello, Andrea Baronchelli.
- **Link:** [Science Advances (2025)](https://www.science.org/doi/10.1126/sciadv.adu9368) / [arXiv:2407.10860](https://arxiv.org/abs/2407.10860)

### **12. Jimenez-Romero et al., 2025**

- **Title:** *Multi-agent systems powered by large language models: applications in swarm intelligence*
- **Authors:** Cristian Jimenez-Romero, Alper Yegenoglu, Christian Blum.
- **Link:** [Frontiers in Artificial Intelligence](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1507742/full) / [arXiv:2501.06322](https://arxiv.org/abs/2501.06322)

### **13. Tran et al., 2025**

- **Title:** *Multi-Agent Collaboration Mechanisms: A Survey of LLMs*
- **Authors:** Khanh-Tung Tran, Dung Dao, Minh-Duong Nguyen, Quoc-Viet Pham, Barry O'Sullivan, Hoang D. Nguyen.
- **Link:** [arXiv:2501.06322](https://arxiv.org/abs/2501.06322)

Yes! **Absolutely** - this is one of the most important experiments in the paper! Let me break down exactly what they did:

## The RL Fine-Tuning Experiments (Section: "Reinforcement Learning Experiments")

### Experimental Setup

They trained models on the **Countdown task** (arithmetic puzzle) using **3 conditions**:

1. **Baseline (RL only)**: Qwen-2.5-3B base model, no prior fine-tuning
2. **Conversation fine-tuning → RL**: First fine-tune on multi-agent dialogues, then RL
3. **Monologue fine-tuning → RL**: First fine-tune on step-by-step reasoning, then RL

### How They Generated Conversational Training Data

They used **Qwen-2.5-32B-IT** to generate multi-agent dialogues for 8,262 reasoning tasks:

**Format (from Supplementary Table 7):**

```
<cast_of_characters>
<persona1> Extrovert mathematician focused on arithmetic heuristics. </persona1>
<persona2> Analytical engineer emphasizing step efficiency. </persona2>
</cast_of_characters>

<conversation>
<think1> Let's first compute 30 − 25 = 5 to simplify the target space. </think1>
<think2> That yields 5, we can now multiply by 4 to approach 20. </think2>
<think1> Good idea. 5 × 4 = 20, but we need 32. </think1>
<think2> Wait, let me recalculate... </think2>
</conversation>

<group_consensus> The best sequence is (30 − 25 + 3) × 4 = 32. </group_consensus>

```

They generated dialogues with **2, 3, or 4 personas** and sampled **600 examples** that reached correct answers (500 training, 100 validation).

### The Monologue Data (Comparison)

For the **same 600 problems**, they generated standard chain-of-thought:

```
<think>
To reach 32 from {25, 30, 3, 4}, I'll try combining operations.
30 − 25 = 5. Then 5 + 3 = 8. Finally, 8 × 4 = 32.
Let me verify: (30 − 25 + 3) × 4 = 8 × 4 = 32. Correct.
</think>
<answer> (30 − 25 + 3) × 4 = 32 </answer>

```

**Key**: Same problems, same correct answers, but different *reasoning structure*.

### The Results (This is Wild!)

**From Extended Data Fig. 8:**

**Qwen-2.5-3B Results:**

- At **step 40**:
    - Conversation-primed: ~38% accuracy
    - Monologue-primed: ~28% accuracy
    - Baseline: ~15% accuracy

**Llama-3.2-3B Results (even more dramatic):**

- At **step 70**:
    - Conversation-primed: ~11% accuracy
    - Monologue-primed: ~5% accuracy
- At **step 150**:
    - Conversation-primed: **40% accuracy**
    - Monologue-primed: **18% accuracy** (less than half!)

### The Spontaneous Emergence Result (Fig. 4)

Even **without** conversational fine-tuning, the baseline model **spontaneously develops** conversational behaviors when rewarded only for accuracy:

**From Fig. 4b** - During RL training, baseline model shows:

- Question & Answering: rises dramatically
- Perspective Shifts: emerges around step 80
- Conflict of Perspectives: emerges around step 100
- Reconciliation: slight increase

**The personalities that emerged** (Fig. 4e):

- **Perspective 1** (Step 120): Methodical problem-solver (High C, Low O)
- **Perspective 2** (Step 120): Trial-and-error explorer (High O, High E)
- **Meta-cognitive monitor**: Reflects on solvability (High N)

The model literally starts talking to itself with distinct personas!

### Cross-Domain Transfer (Supplementary Methods)

They also tested if conversational scaffolding **transfers across domains**:

- Fine-tune on **Countdown** (arithmetic) with conversational format
- Test on **PolitiFact misinformation detection** (completely different!)
- Result: Conversation-primed models learn faster on the new domain too!

### The Prompts They Used (Supplementary Methods)

**For generating conversational data:**

```
You are simulating a collaborative group of thinkers solving a problem.
Each thinker has a distinct persona and engages in a realistic,
back-and-forth conversation. Thinkers may speak in any order and as
many times as needed—no fixed turn order required.

Do not try to be overly positive or polite during the conversation;
focus on puzzle-solving, and note that disagreements can be helpful
for the reasoning.

Assume that there are 2/3/4 thinkers and follow exactly the tag
structure below...

```

## Why This Matters

This experiment proves that:

1. **Conversational scaffolding accelerates learning** - conversation-primed models reach higher accuracy faster
2. **It's not about the answers** - both conditions saw the same correct solutions, but conversation format helps
3. **Models spontaneously develop multi-agent thinking** when rewarded for accuracy alone
4. **The structure generalizes** across domains (arithmetic → misinformation)

So yes, **they absolutely did this** - and it's arguably the most important causal evidence in the paper that social/conversational structure *drives* reasoning improvement, not just correlates with it!
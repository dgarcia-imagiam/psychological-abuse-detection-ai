# PADAI: Psychological Abuse Detection AI
> **Where Generative AI meets gender-based violence prevention.**  
> *Using state-of-the-art language models to uncover hidden psychological abuse in everyday messages.*

## Mission & Social Context

Psychological abuse often hides in plain sight within everyday communication. In messages between former partners, words that appear caring or casual can mask manipulative, controlling, or degrading intent. Unlike explicit slurs or threats, these implicit forms of abuse are harder to recognize but can be just as damaging—eroding a survivor’s self-esteem and well-being over time while leaving little tangible evidence. Detecting this hidden abuse matters because it validates survivors’ experiences, provides crucial context for legal or counseling support, and shines a light on a form of gender-based violence that frequently goes unaddressed.

PADAI (Psychological Abuse Detection AI) is dedicated to tackling this challenge by uncovering subtle signals of emotional abuse in text. It leverages state-of-the-art language models trained on a unique corpus of real, anonymized messages from women who have experienced psychological abuse, ensuring the AI learns from genuine patterns of manipulation and gaslighting found in real life. In practical terms, PADAI can serve as an early-warning and evidence-gathering tool: it might flag harmful patterns in a survivor’s chat history, help lawyers and advocates identify abusive content for legal cases, or assist counselors in understanding the dynamics at play in a survivor’s story. By bringing these hidden patterns to light, the system empowers those affected and those supporting them to intervene earlier and more effectively.

For the long term, PADAI is also a platform for research across disciplines. Its development and dataset open new avenues—from computational social science studies on how abuse manifests in communication, to improving how AI language models grasp nuanced human behavior, to examining the ethics of deploying AI in sensitive, high-stakes contexts. The project is designed to be accessible and impactful for a broad community—from legal professionals, NGOs, and journalists documenting abuse, to survivors seeking validation, to researchers in technology and ethics. By bridging cutting-edge AI with lived experiences, PADAI offers a responsible, purpose-driven approach to technology for social good.

## Key Features

- **Open-Science Seed Dataset ⚙️ (alpha)**  
  An anonymized, ethically curated starter corpus of Spanish-language messages for evaluating psychological-abuse detection—released under an open licence to enable independent benchmarking and reuse. A larger public release will follow as we expand the dataset with new cases and complete further safeguarding reviews.

- **LangChain Detection Pipelines ✅ (prototype)**  
  AI workflows built with LangChain, designed to detect subtle and implicit signs of psychological abuse in written communication—capturing context and nuance beyond simple keyword spotting.

- **Prompt Engineering Library ✏️ (prototype)**  
  Rigorously tested, bias-audited prompt templates that boost recall of implicit abuse while keeping false positives low; easily plug-and-play in other LLM stacks.

- **Ethical & Legal Framework 🛠️ (draft)**  
  Co-developed with legal and clinical experts, this living outline covers privacy, protection for those affected by psychological abuse, and responsible-AI principles. A fuller guideline set will be published as the project’s data and tooling expand.

- **Interdisciplinary Collaboration 🎯 (active)**  
  Built to foster collaboration among experts in legal, psychological, technical, and advocacy fields. This cross-sector approach ensures the tool benefits from diverse perspectives and addresses real-world needs.

- **Self-Help LLM Guidance 📚 (planned)**  
  A plain-language handbook that teaches those affected by psychological abuse how to use open LLM tools (e.g., ChatGPT) to:  
    - analyse toxic or manipulative messages safely  
    - craft calm, boundary-setting responses  
    - identify when to seek professional or legal help
  
  Includes vetted, bias-checked prompt templates and best-practice privacy tips.


> *We invite contributors and collaborators from all sectors to get involved and help advance PADAI’s mission.*

## Call to Action

Help us take PADAI from prototype to life-saving practice. Whether you write code, report stories, litigate cases, or support survivors directly, there’s a place for you here:

| You are…                             | How you can help                                                                                                                                                              |
| ------------------------------------ |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Survivors & Front-line Advocates** | Share *anonymized* message patterns or lived-experience insights (secure channels available). Pilot the tool in support workflows and tell us what works—and what doesn’t.    |
| **Lawyers & Legal Clinics**          | Evaluate PADAI as an evidence-screening aid, advise on admissibility standards, and co-author best-practice briefs that translate AI findings into courtroom-ready arguments. |
| **Psychologists & Counselors**       | Stress-test our abuse taxonomies, contribute therapeutic perspectives, and help design safe user journeys for clients who review toxic conversations.                         |
| **NGOs & Women’s Shelters**          | Partner on ethically collecting new data, co-run impact studies, or integrate PADAI into intake and safety-planning processes.                                                |
| **Researchers & Data Scientists**    | Fork the code, propose new model architectures, or replicate our benchmarks. We welcome pull requests, issue reports, and joint grant proposals.                              |
| **Journalists & Storytellers**       | Use our findings to highlight the often-invisible face of psychological abuse, or collaborate on data-driven investigations that push the topic into public discourse.        |
| **Funders & Policy Makers**          | Sponsor dataset expansion, community roll-outs, or independent audits. Work with us to shape regulations that protect survivors while fostering responsible innovation.       |


## Who We Are

A multidisciplinary core team combining law, psychology, storytelling, organisational change and advanced AI research.

| Name                                                                    | Background                                                                                                                                                                                                                       |
|-------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **[Laura García Broto](https://www.linkedin.com/in/lauragarciabroto/)** | Award-winning filmmaker, founder & former CEO of an international production company and ex-Volkswagen Group communications strategist—brings narrative craft, public-facing outreach and ethical storytelling to PADAI.         |
| **[Ruth Sala](https://www.linkedin.com/in/ruthsala/)**                  | Criminal lawyer specialised in technology-facilitated abuse; advises courts, police and NGOs on digital evidence and has lectured widely on online maltreatment. She grounds the project’s outputs in real-world legal standards.|
| **[Chelo Fernández](https://www.linkedin.com/in/ruthsala/)**            | Digital-transformation and Agile-culture strategist, university lecturer and advocate for human-centred tech adoption. She steers PADAI’s change-management and empathy-by-design practices.                                     |
| **[David García Broto](https://www.linkedin.com/in/davidgarciabroto/)** | Applied-AI scientist specialising in large language models, RAG pipelines and multi-agent systems; lead architect for PADAI’s model design, evaluation and open-source tooling.                                                  |

> Together we combine investigative storytelling, legal rigor, organizational change expertise, entrepreneurial execution and cutting-edge generative-AI research—laying the foundation for a tool that is technically robust, legally sound and socially impactful.

## License

PADAI’s source code and dataset are released under the [Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International](LICENSE) license.

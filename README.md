<div align="center">

## VisTR: Visualizations as Representations for Time-series Table Reasoning
</div>

<div align="center" style="display: flex; justify-content: center; align-items: center; gap: 20px;">

<a href="https://arxiv.org/abs/2406.03753" style="display: flex; align-items: center;">
  <img src="https://img.shields.io/badge/arxiv-red" alt="arxiv" style="height: 20px; vertical-align: middle;">
</a>

</div>

---

### Abstract
Time-series table reasoning interprets temporal patterns and relationships in data to answer user queries. Despite recent advancements leveraging large language models (LLMs), existing methods often struggle with pattern recognition, context lost in long time-series data, and the lack of visual-based reasoning capabilities. To address these challenges, we propose *VisTR*, a framework that places visualizations at the core of the reasoning process. Specifically, *VisTR* leverages visualizations as representations to bridge raw time-series data and human cognitive processes. By transforming tables into fixed-size visualization references, it captures key trends, anomalies, and temporal relationships, facilitating intuitive and interpretable reasoning. These visualizations are aligned with user input, *i.e.*, charts, text, and sketches, through a fine-tuned multimodal LLM, ensuring robust cross-modal alignment. To handle large-scale data, *VisTR* integrates pruning and indexing mechanisms for scalable and efficient retrieval. Finally, an interactive visualization interface supports seamless multimodal exploration, enabling users to interact with data through both textual and visual modalities. Quantitative evaluations demonstrate the effectiveness of *VisTR* in aligning multimodal inputs and improving reasoning accuracy. Case studies further illustrate its applicability to various time-series reasoning and exploration tasks. 

---

### Project Structure
- **Model/**: Contains the trained model files.
- **Training Dataset/**: Contains (chart, sketch, text) pairs used for model training.
- **User Study/**: Contains evaluation results of chart-sketch pairs, as well as user evaluations of the interface.

---

### Citation

If this work is helpful, please kindly cite as:

```bibtex
@misc{hao2024vistr,
      title={VisTR: Visualizations as Representations for Time-series Table Reasoning}, 
      author={Jianing Hao and Zhuowen Liang and Chunting Li and Yuyu Luo and Jie Li and Wei Zeng},
      year={2024},
      eprint={2406.03753},
      archivePrefix={arXiv},
      primaryClass={cs.HC},
      url={https://arxiv.org/abs/2406.03753}, 
}
```
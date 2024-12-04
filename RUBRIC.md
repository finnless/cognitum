**Title:** Build open-text social survey response coder library
**Team:** Nolan
**Project Description:** This project is an open-source library that uses language models to code open-ended survey responses. It is designed to be accessible to social science researchers. Features include setting a codebook, inputting survey responses, and validating classifier accuracy with labeled test data.

**Rubric:**

| **Percent of Grade** | **Task Description** |
|----------------------|----------------------|
| **15%** | **Project Setup**<br> - Create a `README.md` with instructions for setting up the environment, installing dependencies, and running the model. The setup must be clear and reproducible and include example use cases. |
| **25%** | **Model Implementation**<br> - Implement the core model class for text classification. It processes datasets and returns predictions with confidence scores. |
| **20%** | **Data Loading**<br> - Develop a dataset loading feature to simplify data loading and sampling. It must handle both valid and invalid data inputs. |
| **15%** | **Evaluation**<br> - Build evaluation pipeline that tests library performance. Run tests into a CI pipeline for automatic testing on new commits. |
| **10%** | **Error Handling**<br> - Implement error handling and validation for the model and dataset components. Exceptions must be raised and handled gracefully. |
| **5%** | **Documentation**<br> - Maintain clear documentation throughout the codebase, including docstrings and inline comments. |
| **10%** | **Publicizing the Project**<br> - Share the project by posting to pypi, writing a blog post, posting to Hacker News, and sharing on r/LocalLlama. Include a project description and usage instructions in each post. |
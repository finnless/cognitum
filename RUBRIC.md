**Title:** Build open-text social survey response coder library
**Team:** Nolan
**Project Description:** This project is a open source library that uses language models to code open-ended response questions often used in public opinion polling. The target audience for this project social science researches so the library will be easy to use. There will be features for setting a codebook, inputing survey responses, and validating the classifiers accuracy on manually label ground truth test data.

**Rubric:**

| **Percent of Grade** | **Task Description** |
|----------------------|----------------------|
| **15%** | **Project Setup and Environment Configuration**<br> - Develop a comprehensive `README.md` that includes clear instructions for setting up the environment, installing dependencies, and running the model. Ensure the setup process is smooth and reproducible. |
| **25%** | **Model Implementation and Functionality**<br> - Implement the core model class to handle text classification tasks. Ensure the model can process datasets and return accurate predictions with or without confidence scores. |
| **20%** | **Dataset Management**<br> - Develop a robust dataset manager to handle data loading, sampling, and hashing. Ensure the manager can efficiently process both valid and invalid data inputs. |
| **15%** | **Experimental Component and CI/CD**<br> - Design and conduct experiments to evaluate the model's performance. This includes setting up test cases, running the model, and collecting results. Integrate these tests into a CI/CD pipeline to ensure all functionality is tested automatically on new commits. |
| **10%** | **Error Handling and Validation**<br> - Implement comprehensive error handling and validation mechanisms for both the model and dataset components. Ensure that appropriate exceptions are raised and handled gracefully. |
| **5%** | **Documentation and Code Quality**<br> - Maintain high-quality documentation throughout the codebase. This includes clear docstrings and inline comments. |
| **10%** | **Publicizing and Sharing the Project**<br> - Share the project by writing a blog post, posting to Hacker News, and sharing on r/LocalLlama. Include a detailed project description and usage instructions in each post. |
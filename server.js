const express = require("express");
const { PythonShell } = require("python-shell");
const app = express();
const port = 3000;

app.use(express.json());

// Log incoming requests
app.use((req, res, next) => {
  console.log(`Incoming request: ${req.method} ${req.url}`);
  console.log("Request body:", req.body);
  next();
});

app.post("/recommend", (req, res) => {
  console.log("Received POST request at /recommend");

  const ingredients = req.body.ingredients.join(", ");
  console.log("Formatted ingredients:", ingredients);

  const options = {
    pythonPath: "python",
    scriptPath: ".",
    args: [ingredients],
  };

  console.log("Executing Python script with options:", options);

  PythonShell.run("recommender_model.py", options, (err, result) => {
    if (err) {
      console.error("Python script execution error:", err);
      res.status(500).json({ error: "An error occurred" });
    } else {
      console.log("Python script execution completed successfully");
      const recommendations = JSON.parse(result);
      console.log("Recommendations:", recommendations);
      res.json(recommendations);
    }
  });
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});

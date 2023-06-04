const express = require("express");
const { PythonShell } = require("python-shell");
const app = express();
const port = 3000; // or any other available port

// Define an API endpoint for receiving input and sending recommendations
app.get("/recommend", (req, res) => {
  const ingredients = req.query.ingredients; // Get the input ingredients from the query string
  console.log("Received request with ingredients:", ingredients);

  // Set the options for the Python script execution
  const options = {
    pythonPath: "python", // Modify the path if needed (e.g., 'python3')
    scriptPath: ".", // Set the path to the directory containing your Python script
    args: [ingredients],
  };

  // Execute the Python script using PythonShell
  console.log("Executing Python script...");
  PythonShell.run("recommender_model.py", options, (err, result) => {
    if (err) {
      console.error("Error occurred while executing Python script:", err);
      res.status(500).json({ error: "An error occurred" });
    } else {
      console.log("Python script execution completed successfully");
      const recommendations = JSON.parse(result); // Parse the JSON result from Python
      console.log("Recommendations:", recommendations);
      res.json(recommendations);
    }
  });
});

// Start the server
app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});

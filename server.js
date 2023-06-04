const express = require("express");
const { spawn } = require("child_process");
const app = express();
const port = 3000;

app.use(express.json());

app.post("/recommend", (req, res) => {
  const ingredients = req.body.ingredients.join(", ");

  const pythonProcess = spawn("python", ["recommender_model.py", ingredients]);

  pythonProcess.stdout.on("data", (data) => {
    const recommendations = JSON.parse(data.toString());
    res.json(recommendations);
  });

  pythonProcess.stderr.on("data", (data) => {
    console.error(`Python script execution error: ${data}`);
    res.status(500).json({ error: "An error occurred" });
  });
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});

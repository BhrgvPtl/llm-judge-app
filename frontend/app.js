const API_BASE = "http://localhost:8000";

const taskSelect = document.getElementById("task-select");
const promptInput = document.getElementById("prompt-input");
const runButton = document.getElementById("run-button");
const statusDiv = document.getElementById("status");
const finalAnswerPre = document.getElementById("final-answer");
const metaInfoPre = document.getElementById("meta-info");
const candidatesDiv = document.getElementById("candidates");

async function fetchTasks() {
  try {
    statusDiv.textContent = "Loading tasks...";
    const res = await fetch(`${API_BASE}/tasks`);

    if (!res.ok) {
      const text = await res.text();
      console.error("Failed to fetch /tasks:", res.status, text);
      statusDiv.textContent = `Error loading tasks: ${res.status}`;
      return;
    }

    const data = await res.json();
    console.log("Tasks response:", data);

    const tasks = data.tasks || [];

    taskSelect.innerHTML = "";

    if (!tasks.length) {
      statusDiv.textContent = "No tasks returned from backend.";
      return;
    }

    tasks.forEach((t) => {
      const opt = document.createElement("option");
      opt.value = t.id;
      opt.textContent = t.label;
      taskSelect.appendChild(opt);
    });

    statusDiv.textContent = "";
  } catch (err) {
    console.error("Error fetching tasks:", err);
    statusDiv.textContent = "Error fetching tasks (check console).";
  }
}

async function runModels() {
  const task = taskSelect.value;
  const prompt = promptInput.value.trim();

  if (!prompt) {
    alert("Please enter a prompt.");
    return;
  }

  if (!task) {
    alert("Please select a task.");
    return;
  }

  statusDiv.textContent = "Running models and peer reviews...";
  finalAnswerPre.textContent = "";
  metaInfoPre.textContent = "";
  candidatesDiv.innerHTML = "";

  try {
    const res = await fetch(`${API_BASE}/solve`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ task, prompt }),
    });

    if (!res.ok) {
      const text = await res.text();
      console.error("Error from /solve:", res.status, text);
      statusDiv.textContent = `Error from backend: ${res.status}`;
      return;
    }

    const data = await res.json();
    console.log("Solve response:", data);

    finalAnswerPre.textContent = data.final_answer || "";

    if (data.meta) {
      metaInfoPre.textContent = JSON.stringify(data.meta, null, 2);
    }

    const candidates = data.candidates || [];
    candidates.forEach((c) => {
      const card = document.createElement("div");
      card.className = "candidate-card";

      const header = document.createElement("div");
      header.className = "candidate-header";
      header.textContent = `${c.model_id} (candidate #${c.candidate_id})`;
      card.appendChild(header);

      const answerPre = document.createElement("pre");
      answerPre.className = "candidate-answer";
      answerPre.textContent = c.text;
      card.appendChild(answerPre);

      const scoresPre = document.createElement("pre");
      scoresPre.className = "candidate-scores";
      scoresPre.textContent = `Peer scores:\n${JSON.stringify(
        c.peer_scores,
        null,
        2
      )}\n\nPeer explanations:\n${JSON.stringify(
        c.peer_explanations,
        null,
        2
      )}`;
      card.appendChild(scoresPre);

      candidatesDiv.appendChild(card);
    });

    statusDiv.textContent = `Done. ${candidates.length} candidates evaluated.`;
  } catch (err) {
    console.error("Error calling /solve:", err);
    statusDiv.textContent = "Error calling backend (check console).";
  }
}

runButton.addEventListener("click", runModels);

// Fetch tasks on page load
fetchTasks();

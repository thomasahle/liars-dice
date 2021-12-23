const historyDiv = document.getElementById("history-div");
const newButton = document.getElementById("new-button");
const sizeInput = document.getElementById("size-input");
const rollSpan = document.getElementById("roll");
const robotDiceSpan = document.getElementById("robot-dice");
const callBox = document.getElementById("call-box");
const lieLink = document.getElementById("lie-link");

let Ds = [1, 1];
const SIDES = 6;
let D_PRI = Math.max(Ds[0], Ds[1]) * SIDES + 2;
let PRI_INDEX = D_PRI - 2;
let N_ACTIONS = (Ds[0] + Ds[1]) * SIDES + 1;
let CUR_INDEX = N_ACTIONS;
let D_PUB_PER_PLAYER = N_ACTIONS + 1;
let D_PUB = 2 * D_PUB_PER_PLAYER;

let rs = [[], []];
let last_call = -1;
let humanId = 0;
let privs = [null, null];
let state = null;

let DEFAULT_SIZE = 3;
let MAX_SIZE = 4;

const phrases = [
   "I'll say",
   "Maybe",
   "What about",
   "I guess",
   "OK,",
   "Then I say",
   "I say",
   "Aha,",
   "Hmm,"
]

let session = {};

////////////////////////////////////////////////////////////////////////////////
// Model stuff
////////////////////////////////////////////////////////////////////////////////

async function value(state, priv) {
   const res = await session[Ds].run({ priv: priv, pub: state });
   return res.value.data[0];
}

// Load our model.
async function main() {
   newButton.addEventListener("click", newGameClicked);
   lieLink.addEventListener("click", () => submit(N_ACTIONS - 1));
   sizeInput.setAttribute("max", MAX_SIZE);
   sizeInput.setAttribute("value", DEFAULT_SIZE);

   await newGame(DEFAULT_SIZE, DEFAULT_SIZE, -1);
}

main();

async function newGame(D1, D2, newHumanId) {
   Ds = [D1, D2];
   D_PRI = Math.max(Ds[0], Ds[1]) * SIDES + 2;
   PRI_INDEX = D_PRI - 2;

   N_ACTIONS = (Ds[0] + Ds[1]) * SIDES + 1;
   CUR_INDEX = N_ACTIONS;
   D_PUB_PER_PLAYER = N_ACTIONS + 1;
   D_PUB = 2 * D_PUB_PER_PLAYER;

   empty(historyDiv);
   if (!(Ds in session)) {
      let path = "./model_" + Ds[0] + "" + Ds[1] + "_joker.onnx";
      console.log("Loading model " + path);
      addStringToHistory("Loading brain...");
      session[Ds] = await ort.InferenceSession.create(path);
      empty(historyDiv);
      console.log("Done.");
   }

   if (newHumanId === -1) {
      humanId = Math.floor(Math.random() * 2);
   } else {
      humanId = newHumanId;
   }
   console.log("Human id: " + humanId);

   last_call = -1;

   for (let p = 0; p < 2; p++) {
      rs[p].length = 0; // Clear previous roll
      privs[p] = new ort.Tensor(
         "float32",
         new Float32Array(Array(D_PRI).fill(0))
      );
      privs[p].data[PRI_INDEX + p] = 1;

      for (let i = 0; i < Ds[p]; i++) {
         const r = Math.floor(Math.random() * SIDES);
         rs[p].push(r + 1);
      }
      for (let face = 1; face <= SIDES; face++) {
         let cnt = 0;
         for (let i = 0; i < Ds[p]; i++)
            if (rs[p][i] === face)
               cnt += 1;
         for (let i = 0; i < cnt; i++) {
            privs[p].data[(face-1) * Math.max(D1, D2) + i] = 1
         }
      }

      rs[p].sort();
   }
   state = new ort.Tensor("float32", new Float32Array(Array(D_PUB).fill(0)));
   state.data[CUR_INDEX] = 1;

   empty(rollSpan);
   empty(robotDiceSpan);
   for (let i = 0; i < Ds[humanId]; i++) {
      rollSpan.appendChild(newDiceIcon(rs[humanId][i]));
   }
   for (let i = 0; i < Ds[1-humanId]; i++) {
      const elem = document.createElement("i");
      elem.className = "bi-question-square";
      elem.classList.add("small-dice");
      robotDiceSpan.appendChild(elem);
   }

   callBox.scrollTop = 0;

   lieLink.classList.add('hidden');

   empty(callBox);
   for (let i = 1; i <= Ds[0]+Ds[1]; i++) {
      for (let f = 1; f <= SIDES; f++) {
         const div = document.createElement("div");
         div.className = "dice-button";
         div.appendChild(document.createTextNode(i + " "));
         div.appendChild(newDiceIcon(f));
         callBox.appendChild(div);
         const action = (i - 1) * SIDES + (f - 1);
         div.id = 'action-' + action;
         div.addEventListener("click", () => submit(action));
      }
   }


   if (humanId === 0)
      addStringToHistory("Make your bid!");
   else
      addStringToHistory("The robot starts guessing...");

   if (humanId !== 0) await goRobot();
}

async function newGameClicked() {
   const n = Number.parseInt(sizeInput.value, 10);
   if (n === undefined || n > MAX_SIZE || n < 1) {
      console.log("Unsupported size", sizeInput.value);
   } else {
      // We swap the player starting every time
      await newGame(n, n, 1-humanId);
   }
}

function newDiceIcon(i) {
   const elem = document.createElement("i");
   elem.className = "bi-dice-" + i;
   elem.classList.add("small-dice");
   return elem;
}


function clickDice(elem, i) {
   for (let i = 1; i <= SIDES; i++) {
      const elem = document.getElementById("dice-"+i);
      elem.classList.remove('clicked');
   }
   elem.classList.add('clicked');
}



function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function empty(element) {
   while (element.firstChild)
      element.removeChild(element.firstChild);
}

async function addElementToHistory(elem, class_="standard") {
   const para = document.createElement("div");
   para.classList.add("new-call");
   para.classList.add(class_);
   para.appendChild(elem);
   historyDiv.appendChild(para);

   scrollBox(historyDiv, historyDiv.scrollHeight, 500);

   await sleep(500);
}

async function addStringToHistory(string, class_) {
   const text = document.createTextNode(string);
   await addElementToHistory(text, class_);
}


function actionToSpan(prefix, action, postfix) {
   const span = document.createElement("span");
   if (action === N_ACTIONS - 1) {
      span.appendChild(document.createTextNode(prefix + "liar!" + postfix));
   } else {
      const n = Math.floor(action / SIDES) + 1;
      const d = (action % SIDES) + 1;

      // Drop the long form "times"
      //span.appendChild(document.createTextNode(prefix + n + " times "));
      span.appendChild(document.createTextNode(prefix + n + " "));
      span.appendChild(newDiceIcon(d));
      span.appendChild(document.createTextNode(postfix));
   }
   return span
}

async function submit(action) {
   if (action == N_ACTIONS-1) {
      _apply_action(state, action);
      const oldCall = last_call;
      last_call = action;
      lieLink.classList.add('hidden');
      await addElementToHistory(actionToSpan("", action, ""), 'human-call');
      endGame(oldCall, false);
      return;
   }

   if (action <= last_call) {
      console.log("Call is too low");
      return;
   }

   _apply_action(state, action);
   const oldCall = last_call;
   await setLastCall(action);

   lieLink.classList.add('hidden');
   await addElementToHistory(actionToSpan("🐵: ", action, ""), 'human-call');
   await goRobot();
}

async function goRobot() {
   const action = await sampleCallFromPolicy();
   _apply_action(state, action);
   const oldCall = last_call;
   await setLastCall(action);
   const prefix = phrases[Math.floor(Math.random() * phrases.length)];
   await addElementToHistory(actionToSpan("🤖: "+prefix+" " , action, ""), 'robot-call');
   if (action === N_ACTIONS - 1) {
      endGame(oldCall, true);
   }
   else {
      lieLink.classList.remove('hidden');
   }
}

function scrollBox(elem, diff, duration) {
   var startingY = elem.scrollTop;
   var to = startingY + diff;
   // Just some monkey patching
   elem.currentScrollingDestination = to;

   var start;
   // Bootstrap our animation - it will get called right before next frame shall be rendered.
   window.requestAnimationFrame(function step(timestamp) {
      // Somebody else is doing the scrolling now
      if (elem.currentScrollingDestination !== to) {
         return;
      }
      if (!start) {
         start = timestamp;
      }
      var elapsed = timestamp - start;
      var percent = Math.min(elapsed / duration, 1);
      elem.scrollTop = startingY + diff * percent;
      if (elapsed < duration) {
         window.requestAnimationFrame(step);
      }
   })
}

async function setLastCall(action) {
   for (let i = last_call + 1; i <= Math.min(action, N_ACTIONS-2); i++) {
      const button = document.getElementById("action-"+i);
      button.classList.add("gray");
   }

   // Actually update last_call
   last_call = action;

   // Make sure the next good element is visible, if any are left
   if (action+1 < N_ACTIONS-1) {
      const boxRect = callBox.getBoundingClientRect();
      const buttonRect = document.getElementById("action-"+(action+1)).getBoundingClientRect();
      scrollBox(callBox, buttonRect.top - boxRect.top, 300);
   }
}

function endGame(call, isRoboCall) {
   const n = Math.floor(call / SIDES) + 1;
   const d = (call % SIDES) + 1;

   let actual = 0;
   for (let p = 0; p < 2; p++) {
      for (let i = 0; i < rs[p].length; i++) {
         if (rs[p][i] === d || rs[p][i] === 1) actual += 1;
      }
   }
   const isGood = actual >= n;
   addElementToHistory(actionToSpan("The call \"", call, "\" was " + isGood + "!"));
   addElementToHistory(actionToSpan("There were ", (actual-1)*SIDES+d-1, "s in total."));

   // Reveal robot dice
   empty(robotDiceSpan);
   for (let i = 0; i < Ds[1-humanId]; i++) {
      robotDiceSpan.appendChild(newDiceIcon(rs[1-humanId][i]));
   }
   document.querySelectorAll("#hands .bi-dice-1, #hands .bi-dice-"+d).forEach(icon => icon.classList.add("highlight"));


   // Reduce number of dice for winner
   let newDs = [...Ds];
   const robotWon = (!isRoboCall && isGood) || (isRoboCall && !isGood);
   if (robotWon) {
      newDs[1 - humanId] -= 1;
   } else {
      newDs[humanId] -= 1;
   }

   // If we continue for more rounds
   if (newDs[0] > 0 && newDs[1] > 0) {
      if (robotWon) {
         addStringToHistory("🤖 wins the round!");
      } else {
         addStringToHistory("🎉 You win the round!");
      }

      const lastLine = createLastLine("Continue...");
      lastLine.addEventListener("click", () => {
         // We Make it so the loser always starts.
         // This only has the negative side that we'll never get to
         // a 3 vs 2 game, say, where the 2 dice player goes first.
         let newHumanId = robotWon ? 0 : 1;
         if (newHumanId !== humanId)
            newDs = [newDs[1], newDs[0]];
         newGame(newDs[0], newDs[1], newHumanId);
      });
      addElementToHistory(lastLine);
   }
   // Game over
   else {
      if (robotWon) {
         addStringToHistory("🤖 wins the game!");
      } else {
         addStringToHistory("🎉 You win the game!");
      }

      const newGameLine = createLastLine("New Game");
      newGameLine.addEventListener("click", newGameClicked);
      addElementToHistory(newGameLine);
   }
}

function createLastLine(text) {
   const newGameLine = document.createElement("div");
   newGameLine.className = "last-line";
   newGameLine.appendChild(document.createTextNode("🎲 "));
   const link = document.createElement("span");
   link.className = "link";
   link.appendChild(document.createTextNode(text));
   newGameLine.appendChild(link);
   // Insert an emsp to balance the emoji in the text centering
   newGameLine.appendChild(document.createTextNode(" \u2003"));
   return newGameLine;
}

// Game functions

function get_cur(state) {
   const cur = 1 - state.data[CUR_INDEX];
   if (state.data[CUR_INDEX + D_PUB_PER_PLAYER] !== cur)
      console.log("Warning: Bad current indicator");
   return cur;
}

function _apply_action(state, action) {
   const cur = get_cur(state);
   state.data[action + cur * D_PUB_PER_PLAYER] = 1;
   state.data[CUR_INDEX + cur * D_PUB_PER_PLAYER] = 0;
   state.data[CUR_INDEX + (1 - cur) * D_PUB_PER_PLAYER] = 1;
}

function weightedChoice(array, dist) {
   let r = Math.random();
   return array.find((e, i) => (r -= dist[i]) < 0);
}

async function sampleCallFromPolicy() {

   console.log("State:", [... state.data]);
   console.log("Private:", [... privs[1-humanId].data]);

   const n_actions = N_ACTIONS - last_call - 1;
   const v = await value(state, privs[1 - humanId]);

   const actions = [];
   const regrets = [];
   let sum = 0;
   for (let i = 0; i < n_actions; i++) {
      const clonedState = new ort.Tensor("float32", new Float32Array([...state.data]));
      const action = i + last_call + 1;
      actions.push(action);
      _apply_action(clonedState, i + last_call + 1);
      const vi = await value(clonedState, privs[1 - humanId]);
      const r = Math.max(vi - v, 0);
      regrets.push(r);
      sum += r;
   }

   if (sum === 0) {
      for (let i = 0; i < n_actions; i++) {
         regrets[i] = 1 / n_actions;
      }
   } else {
      for (let i = 0; i < n_actions; i++) {
         regrets[i] /= sum;
      }
   }

   console.log("Probabilities:", regrets);

   return weightedChoice(actions, regrets);
}

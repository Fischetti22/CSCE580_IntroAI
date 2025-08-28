# AI-Powered Passcode Cracking Game

An intelligent passcode guessing game that combines AI analysis with statistical data to crack 4-digit passcodes. The game uses OpenAI's GPT model to analyze previous attempts and suggest optimal guessing strategies.

## 🎮 How It Works

The game challenges you to crack a 4-digit passcode (currently set to `3246`) with only 3 attempts per digit. What makes this unique is the AI assistant that analyzes your previous attempts and provides strategic recommendations.

### Game Features

- **AI-Powered Strategy**: Uses OpenAI's GPT-4 model to analyze guess patterns and suggest optimal strategies
- **Statistical Analysis**: Incorporates data about the most common 4-digit PINs from research studies
- **Attempt Logging**: All guesses are logged to CSV for pattern analysis
- **Three-Strike System**: 3 attempts per digit before lockout
- **Progressive Difficulty**: Must crack each digit sequentially

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- OpenAI API key
- Required Python packages (see Installation)

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Install required packages:
```bash
pip install openai pandas python-dotenv
```

3. Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
```

4. Run the game:
```bash
python passcode.py
```

## 🎯 Gameplay

1. **AI Analysis Phase**: The game first analyzes previous attempts (if any) and provides three strategic plans (A, B, C)
2. **Guessing Phase**: You have 3 attempts to guess each of the 4 digits
3. **Logging**: Each guess is automatically logged for future AI analysis
4. **Win/Lose**: Successfully guess all 4 digits to win, or get locked out after 3 failed attempts per digit

### Example Output
```
Plan A: 1 2 3 4
Plan B: 0 0 0 0  
Plan C: 1 1 1 1

Guess The first digit of the passcode: 3
Correct!
Guess The second digit of the passcode: 2
Correct!
...
```

## 📊 Data Sources

The AI leverages statistical data about common passcode patterns:

- **Most Common PINs**: 1234, 1111, 0000, 1212, 7777, etc.
- **Research Sources**: NY Post and CNBC studies on leaked PIN data
- **Historical Attempts**: Your previous guesses stored in `guesses/guesses_log.csv`

## 🗂️ Project Structure

```
├── passcode.py           # Main game logic
├── most_common.md        # Statistical data about common PINs
├── openai.txt           # OpenAI integration snippets
├── guesses/             # Directory for guess logs
│   └── guesses_log.csv  # Automatically generated guess history
└── .env                 # API keys (create this file)
```

## 🧠 AI Strategy

The AI assistant analyzes:
1. **Historical Performance**: Previous guess patterns and success rates
2. **Statistical Probability**: Most/least common PIN combinations
3. **Strategic Recommendations**: Three-tiered approach (Plan A/B/C)

The system uses temperature=0 for consistent, deterministic responses and provides structured output for easy parsing.

## 🔧 Technical Details

- **Language**: Python 3
- **AI Model**: GPT-4o-mini (fast and cost-effective)
- **Data Storage**: CSV logging with pandas
- **API Integration**: OpenAI Python client
- **Configuration**: Environment variables via python-dotenv

## 📈 Future Enhancements

- [ ] Dynamic passcode generation
- [ ] Difficulty levels
- [ ] Multiplayer support  
- [ ] Advanced pattern recognition
- [ ] Web interface
- [ ] Performance analytics dashboard

## 🤝 Contributing

This project is part of CSCE 580 (Introduction to AI) coursework. Feel free to fork and experiment with different AI strategies or game mechanics.

## 📄 License

This project is for educational purposes as part of university coursework.

## 🎓 Academic Context

**Course**: CSCE 580 - Introduction to Artificial Intelligence  
**Project**: Project A - AI Game Implementation  
**Focus**: Practical application of AI in game strategy and decision-making

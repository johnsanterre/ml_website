(function () {
    const questions = Array.from(document.querySelectorAll('.quiz-question'));
    const resetBtn = document.getElementById('reset-quiz');
    const scoreEl = document.getElementById('quiz-score');

    function checkQuestion(question) {
        const correct = question.dataset.correct;
        const explanations = JSON.parse(question.dataset.explanations || '{}');
        const selected = question.querySelector('input[type="radio"]:checked');
        const feedback = question.querySelector('.quiz-feedback');

        if (!selected) return;

        // Lock this question
        question.querySelectorAll('input[type="radio"]').forEach(r => r.disabled = true);

        // Style each option
        question.querySelectorAll('.quiz-options li').forEach(li => {
            const val = li.querySelector('input[type="radio"]').value;
            li.classList.remove('correct', 'incorrect');
            if (val === correct) li.classList.add('correct');
            else if (val === selected.value) li.classList.add('incorrect');
        });

        // Show feedback
        const isCorrect = selected.value === correct;
        feedback.classList.remove('correct', 'incorrect');
        feedback.classList.add(isCorrect ? 'correct' : 'incorrect');
        feedback.innerHTML = '<strong>' + (isCorrect ? '✓ Correct. ' : '✗ Incorrect. ') + '</strong>'
            + (explanations[selected.value] || '');

        updateScore();
    }

    function updateScore() {
        if (!scoreEl) return;
        const answeredCount = questions.filter(q =>
            q.querySelector('.quiz-feedback.correct, .quiz-feedback.incorrect')
        ).length;
        const correctCount = questions.filter(q =>
            q.querySelector('.quiz-feedback.correct')
        ).length;
        const label = scoreEl.querySelector('.score-label');
        const value = scoreEl.querySelector('.score-value');
        if (answeredCount === questions.length) {
            if (label) label.textContent = 'Score';
            if (value) value.textContent = correctCount + ' / ' + questions.length;
        } else {
            if (label) label.textContent = 'Remaining';
            if (value) value.textContent = (questions.length - answeredCount) + ' / ' + questions.length;
        }
    }

    // Attach per-question change listeners
    questions.forEach(function (question) {
        question.querySelectorAll('input[type="radio"]').forEach(function (radio) {
            radio.addEventListener('change', function () {
                checkQuestion(question);
            });
        });
    });

    // Reset button
    if (resetBtn) {
        resetBtn.addEventListener('click', function () {
            questions.forEach(function (question) {
                question.querySelectorAll('input[type="radio"]').forEach(function (r) {
                    r.disabled = false;
                    r.checked = false;
                });
                question.querySelectorAll('.quiz-options li').forEach(function (li) {
                    li.classList.remove('correct', 'incorrect');
                });
                const fb = question.querySelector('.quiz-feedback');
                if (fb) { fb.className = 'quiz-feedback'; fb.innerHTML = ''; }
            });
            const label = scoreEl && scoreEl.querySelector('.score-label');
            const value = scoreEl && scoreEl.querySelector('.score-value');
            if (label) label.textContent = 'Questions';
            if (value) value.textContent = questions.length;
        });
    }

    // Initialise score display
    updateScore();
})();

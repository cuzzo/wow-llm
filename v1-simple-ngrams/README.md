# Simple N-Gram LLM


## What is an N-Gram?

* "N" just means a number (like 2, or 3, or 4).
* "Gram" means a piece or a part.
* So, an N-Gram is just a tiny piece of text that's 'N' letters long.
    * If N=2 (we call it a "bigram"), "hi", "it", "is" are all N-Grams.
    * If N=3 (a "trigram"), "the", "cat", "sat" are N-Grams.

## How the N-Gram Character Model Writes Text:

1. *Learning Time:* Imagine the computer reads a whole bunch of text, like storybooks or websites. It doesn't understand the stories, but it acts like a detective looking for clues!
   * It looks at short sequences of letters (the N-Grams). For example, if N=3, it looks at every pair of letters (like "th").
   * It keeps a list of which letter came next after each pair. So, it might see "th" followed by "e" lots of times, "th" followed by "a" sometimes, and "th" followed by "x" maybe never. It writes all this down: "After 'th', 'e' is super common!"

2. *Writing Time:* Now you ask the computer to write something. You might give it a starting letter, like "T".
        The computer looks at the last N-1 letters it has written (if N=3, it looks at the last 2 letters). Let's say it has written "Th".
    * It looks at its list: "What letter usually comes after 'Th'?"
    * Its list says "e" is very common. So, it picks "e"! Now it has "The".
    * It repeats! It looks at the last two letters, "he". Checks its list for what comes after "he". Maybe "l", "n", "r" are common. It picks one, maybe "n". Now it has "Then".
    * It keeps doing this: look at the last few letters -> check the list -> pick the most likely next letter -> add it.

## Why it might write "garbage": 

Because the computer only remembers the last few letters (like 2 or 3), it forgets the beginning of the sentence! It just follows the common connections, which can sometimes make sentences that don't make much sense or just repeat words.

## How training works:

We train a "model", which - here - is a simple data structure, a Hash.

Let's imagine we have an input like: "In a land where summers span decades and winters can last a lifetime, trouble is brewing."

If N=3, we will iterate over the input like so:

```
1:

__
In a land where summers span decades and winters can last a lifetime, trouble is brewing.


context="In"
next_char=" "
```

```
2:

 __
In a land where summers span decades and winters can last a lifetime, trouble is brewing.

context="n "
next_char="a"
```

```
3:

  __
In a land where summers span decades and winters can last a lifetime, trouble is brewing.

context=" a"
next_char=" "
```

Eventually, this input will "train" to build a Hash (the model) with this structure:

```
{
  "In": {
    " ": 1
  },
  "n ": {
    "a": 1,
    "d": 1,
    "l": 1
  },
  " a": {
    " ": 2,
    "n": 1
  },
  "a ": {
    "l": 2
  },
  " l": {
    "a": 2,
    "i": 1
  },
  "la": {
    "n": 1,
    "s": 1
  },
  "an": {
    "d": 2,
    " ": 2
  },
  "nd": {
    " ": 2
  },
  "d ": {
    "w": 2
  },
  " w": {
    "h": 1,
    "i": 1
  },
  "wh": {
    "e": 1
  },
  "he": {
    "r": 1
  },
  "er": {
    "e": 1,
    "s": 2
  },
  "re": {
    " ": 1,
    "w": 1
  },
  "e ": {
    "s": 1,
    "i": 1
  },
  " s": {
    "u": 1,
    "p": 1
  },
  "su": {
    "m": 1
  },
  "um": {
    "m": 1
  },
  "mm": {
    "e": 1
  },
  "me": {
    "r": 1,
    ",": 1
  },
  "rs": {
    " ": 2
  },
  "s ": {
    "s": 1,
    "a": 1,
    "c": 1,
    "b": 1
  },
  "sp": {
    "a": 1
  },
  "pa": {
    "n": 1
  },
  " d": {
    "e": 1
  },
  "de": {
    "c": 1,
    "s": 1
  },
  "ec": {
    "a": 1
  },
  "ca": {
    "d": 1,
    "n": 1
  },
  "ad": {
    "e": 1
  },
  "es": {
    " ": 1
  },
  "wi": {
    "n": 2
  },
  "in": {
    "t": 1,
    "g": 1
  },
  "nt": {
    "e": 1
  },
  "te": {
    "r": 1
  },
  " c": {
    "a": 1
  },
  "as": {
    "t": 1
  },
  "st": {
    " ": 1
  },
  "t ": {
    "a": 1
  },
  "li": {
    "f": 1
  },
  "if": {
    "e": 1
  },
  "fe": {
    "t": 1
  },
  "et": {
    "i": 1
  },
  "ti": {
    "m": 1
  },
  "im": {
    "e": 1
  },
  "e,": {
    " ": 1
  },
  ", ": {
    "t": 1
  },
  " t": {
    "r": 1
  },
  "tr": {
    "o": 1
  },
  "ro": {
    "u": 1
  },
  "ou": {
    "b": 1
  },
  "ub": {
    "l": 1
  },
  "bl": {
    "e": 1
  },
  "le": {
    " ": 1
  },
  " i": {
    "s": 1
  },
  "is": {
    " ": 1
  },
  " b": {
    "r": 1
  },
  "br": {
    "e": 1
  },
  "ew": {
    "i": 1
  },
  "ng": {
    ".": 1
  }
}
```

## How generating text works

If we start with an input "an", we will try to pick the most likely third letter.

To do this, we first look up our `current_context` ("an") in our model:

```
  "an": {
    "d": 2,
    " ": 2
  }
```

We have an equal chance of picking "d" and " " (space) next. Let's assume the model picks "d".

The `current_context` is then set to "nd", so it will next look up "nd" in our model:

```
  "nd": {
    " ": 2
  }
```

It has a 100% chance to pick " " (space).

So the `current_context` is set "d ", as we are trying to predict the next character to follow, so we will look up "d " in our model:

```
  "d ": {
    "w": 2
  }
```

Again, we have a 100% chance of picking, "w", so this will set the `current_context` to " w". This is what's in the model:

```
  " w": {
    "h": 1,
    "i": 1
  }
```

Here it has an equal chance of picking either "h" or "i".

By now, hopefully you get the idea of how this works.

## How to run the code

To generate some example output, we can run `./example.rb` with a training input and a prompt:

```
./example.rb "In a land where summers span decades and winters can last a lifetime, trouble is brewing. From the scheming south and the savage eastern lands, to the frozen north and the ancient Wall 1  protecting the realm from the darkness beyond, powerful families are locked in a deadly game for control of the Seven Kingdoms of Westeros.  As betrayal, lust, intrigue, and supernatural forces shake the very foundations of the kingdoms, the honorable Stark family finds itself drawn into the heart of the conflict. Lord Eddard Stark is summoned to serve as the Hand of the King, a position of power that proves more perilous than any battlefield.  Amidst plots and counter-plots, victory and tragedy, alliances and enmities, the fate of the Starks, their allies, and their enemies hangs perilously in the balance as they play the game of thrones, where you win... or you die." "Th"
```

```
ThFvIonedy, the Stark fien a lances, throu whe yountralm the King the the ve Stare entroves cones spand self thronemilound wing the Kingdom fouthe cades shat a decand Severigue, whe gamern a louthang su
```

## Why is this garbage?

Our input is very small.

Since many common 2-letter pairs are not seen in this Hash (and most possible 2-letter pairs), when given random prompts, this model will mostly hit the `fallback`.

Our initial prompt ("Th") was not in the model. The model is case sensitive, so even though "th" had many entries, there were no entries for "Th".

## Fallback

An important concept in LLMs is `fallback`. When the LLM is operating on a context it's never seen before -- "Th" in this case -- it has to do something.

Our current fallback is to generate a random character from the list of seen characters.

In this eample, it picked "F". Since "hF" was also never seen, it picked another random character, and this carried on for some time.

## Can it do better?

Since "th" is in the input data, we can prompt the LLM to generate text using "th" instead of "Th".

```
./example.rb "In a land where summers span decades and winters can last a lifetime, trouble is brewing. From the scheming south and the savage eastern lands, to the frozen north and the ancient Wall 1  protecting the realm from the darkness beyond, powerful families are locked in a deadly game for control of the Seven Kingdoms of Westeros.  As betrayal, lust, intrigue, and supernatural forces shake the very foundations of the kingdoms, the honorable Stark family finds itself drawn into the heart of the conflict. Lord Eddard Stark is summoned to serve as the Hand of the King, a position of power that proves more perilous than any battlefield.  Amidst plots and counter-plots, victory and tragedy, alliances and enmities, the fate of the Starks, their allies, and their enemies hangs perilously in the balance as they play the game of thrones, where you win... or you die." "Th"
```

```
the famin ind or all of plotserilouth ancess battle fame, int of to the and ing sounte a ploter-ploublefing sche shat Wallict. of the ing.  Amilifetronort Wes, prom the kinto their yous control forces a
```

This was not much better. Why?

1) Because we have a VERY small training set.
2) Inevitably, if you pick characters based on probibility, there's a chance you'll end up with words that aren't even plausible.

We want the LLM to be able to generate new possible names it's never seen before, like "Gamonica", but we don't want it to generate something like "Amilifetronort"

## How will we fix this?

Over the next several versions, we will:

1) Trail on larger data sets
2) Move away from character N-Grams to token N-Grams
3) More intelligently pick fallbacks

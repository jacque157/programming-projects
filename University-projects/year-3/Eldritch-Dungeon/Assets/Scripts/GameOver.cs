using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;

public class GameOver : MonoBehaviour
{
    public Image background;
    public Text text;
    public Text infoText;
    public Canvas gui;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if (Player.health <= 0)
        {
            EndScreenText("Game Over", "You succumbed to your wounds, you die. \nPress Esc to quit the game.", Color.red);
        }

        if (Player.sanity <= 0)
        {
            EndScreenText("Game Over", "You could not handle the terrors, you are insane. \nPress Esc to quit the game.", Color.blue);
        }

        CapsuleCollider playersHitbox = GameObject.FindGameObjectWithTag("Player").GetComponent<CapsuleCollider>();

        if (playersHitbox.bounds.Intersects(Map.exitTrigger.bounds))
        {
            EndScreenText("You Win!", "You have managed to escape. \nPress Esc to quit the game.", Color.yellow);
        }
    }


    public void EndScreenText(string textMessage, string infoMessage, Color color)
    {
        background.enabled = true;
        Time.timeScale = 0;
        text.enabled = true;
        gui.enabled = false;
        text.text = textMessage;
        text.color = color;

        infoText.enabled = true;
        infoText.text = infoMessage;
        infoText.color = color;
    }
}

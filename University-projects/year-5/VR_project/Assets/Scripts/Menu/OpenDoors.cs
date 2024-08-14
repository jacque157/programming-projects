using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class OpenDoors : MonoBehaviour
{
    // Start is called before the first frame update
    [SerializeField]
    private Animator a;

    public GameObject uiWelcome;
    void Start()
    {
        a = GetComponent<Animator>();
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void OnPress()
    {
        uiWelcome.SetActive(false);
        a.Play("OpenDoors");
    }
}

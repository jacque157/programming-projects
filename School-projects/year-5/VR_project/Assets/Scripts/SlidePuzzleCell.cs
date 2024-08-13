using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit;

public class SlidePuzzleCell : MonoBehaviour
{
    public SpriteRenderer spriteRenderer;

    public SliderPuzzle sliderPuzzle;
    public XRBaseInteractable interactable;

    private void Start()
    {
        sliderPuzzle = GetComponentInParent<SliderPuzzle>();
        interactable = GetComponent<XRBaseInteractable>();
        interactable.selectExited.RemoveAllListeners();
        interactable.selectExited.AddListener(OnSelectExit);
    }

    public void OnSelectExit(SelectExitEventArgs args)
    {
        sliderPuzzle.CheckCorrectness();
    }

}

use crate::layer_system::{Layer, LayerSystem};
use std::collections::VecDeque;

#[derive(Clone)]
pub struct LayerState {
    pub layers: Vec<Layer>,
    pub selected_layer: Option<usize>,
}

impl LayerState {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            selected_layer: None,
        }
    }

    pub fn from_layer_system(layer_system: &LayerSystem) -> Self {
        Self {
            layers: layer_system.get_layers().to_vec(),
            selected_layer: layer_system.get_selected_layer_id(),
        }
    }
}

pub struct UndoRedoManager {
    history: VecDeque<LayerState>,
    current_index: usize,
    max_history_size: usize,
}

impl UndoRedoManager {
    pub fn new() -> Self {
        Self::with_capacity(100) // Default to 100 undo levels
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(capacity),
            current_index: 0,
            max_history_size: capacity,
        }
    }

    pub fn save_state(&mut self, state: LayerState) {
        // If we're not at the end of history (i.e., we've undone some actions),
        // remove all states after current position
        if self.current_index < self.history.len() {
            self.history.truncate(self.current_index);
        }

        // Add the new state
        self.history.push_back(state);

        // Maintain max history size
        if self.history.len() > self.max_history_size {
            self.history.pop_front();
            self.current_index = self.history.len();
        } else {
            self.current_index = self.history.len();
        }
    }

    pub fn can_undo(&self) -> bool {
        self.current_index > 1
    }

    pub fn can_redo(&self) -> bool {
        self.current_index < self.history.len()
    }

    pub fn undo(&mut self) -> Option<&LayerState> {
        if self.can_undo() {
            self.current_index -= 1;
            self.history.get(self.current_index - 1)
        } else {
            None
        }
    }

    pub fn redo(&mut self) -> Option<&LayerState> {
        if self.can_redo() {
            let state = self.history.get(self.current_index);
            self.current_index += 1;
            state
        } else {
            None
        }
    }

    pub fn clear(&mut self) {
        self.history.clear();
        self.current_index = 0;
    }

    pub fn get_current_state(&self) -> Option<&LayerState> {
        if self.current_index > 0 {
            self.history.get(self.current_index - 1)
        } else {
            None
        }
    }

    pub fn history_size(&self) -> usize {
        self.history.len()
    }

    pub fn current_position(&self) -> usize {
        self.current_index
    }
}

impl Default for UndoRedoManager {
    fn default() -> Self {
        Self::new()
    }
}

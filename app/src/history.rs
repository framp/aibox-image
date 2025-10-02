use image::DynamicImage;

#[derive(Clone, Debug)]
pub enum Action {
    LoadImage { path: String },
    Inpaint { prompt: String },
    Upscale,
    PortraitEdit,
    FaceSwap,
    Selection,
}

impl Action {
    pub fn description(&self) -> String {
        match self {
            Action::LoadImage { path } => format!("Load: {}", path),
            Action::Inpaint { prompt } => format!("Inpaint: {}", prompt),
            Action::Upscale => "Upscale".to_string(),
            Action::PortraitEdit => format!("Portrait Edit"),
            Action::FaceSwap => "Face Swap".to_string(),
            Action::Selection => "Selection".to_string(),
        }
    }
}

#[derive(Clone)]
pub struct HistoryEntry {
    pub action: Action,
    pub image: DynamicImage,
    pub selections: Vec<crate::image_canvas::Selection>,
}

pub struct History {
    entries: Vec<HistoryEntry>,
    current_index: Option<usize>,
    max_entries: usize,
}

impl Default for History {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            current_index: None,
            max_entries: 50,
        }
    }
}

impl History {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            current_index: None,
            max_entries,
        }
    }

    pub fn push(
        &mut self,
        action: Action,
        image: DynamicImage,
        selections: Vec<crate::image_canvas::Selection>,
    ) {
        if let Some(idx) = self.current_index {
            self.entries.truncate(idx + 1);
        } else {
            self.entries.clear();
        }

        self.entries.push(HistoryEntry {
            action,
            image,
            selections,
        });

        if self.entries.len() > self.max_entries {
            self.entries.remove(0);
        }

        self.current_index = if self.entries.is_empty() {
            None
        } else {
            Some(self.entries.len() - 1)
        };
    }

    pub fn can_undo(&self) -> bool {
        self.current_index.map_or(false, |idx| idx > 0)
    }

    pub fn can_redo(&self) -> bool {
        self.current_index
            .map_or(false, |idx| idx < self.entries.len() - 1)
    }

    pub fn undo(&mut self) -> Option<&HistoryEntry> {
        if let Some(idx) = self.current_index {
            if idx > 0 {
                self.current_index = Some(idx - 1);
                return Some(&self.entries[idx - 1]);
            }
        }
        None
    }

    pub fn redo(&mut self) -> Option<&HistoryEntry> {
        if let Some(idx) = self.current_index {
            if idx < self.entries.len() - 1 {
                self.current_index = Some(idx + 1);
                return Some(&self.entries[idx + 1]);
            }
        }
        None
    }

    pub fn goto(&mut self, index: usize) -> Option<&HistoryEntry> {
        if index < self.entries.len() {
            self.current_index = Some(index);
            Some(&self.entries[index])
        } else {
            None
        }
    }

    pub fn current(&self) -> Option<&HistoryEntry> {
        self.current_index.map(|idx| &self.entries[idx])
    }

    pub fn entries(&self) -> &[HistoryEntry] {
        &self.entries
    }

    pub fn current_index(&self) -> Option<usize> {
        self.current_index
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.current_index = None;
    }
}

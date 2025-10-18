use std::borrow::Cow;

use gpui::*;
use gpui_component::{
    ActiveTheme, IconName, Root, TitleBar, blue_400,
    button::{Button, ButtonVariants},
    context_menu::ContextMenuExt,
    green_400,
    group_box::GroupBox,
    h_flex,
    input::{InputState, TextInput},
    label::Label,
    resizable::{ResizableState, h_resizable, resizable_panel},
    sidebar::{Sidebar, SidebarHeader, SidebarMenu, SidebarMenuItem},
    v_flex,
};

mod assets;
mod helper;
mod playground;
mod sidebar;
mod themes;
mod title;

pub use crate::assets::Assets;
pub use crate::playground::Playground;
pub use crate::{sidebar::AppSidebar, title::AppTitleBar};

#[derive(Action, Clone, PartialEq, Eq)]
#[action(namespace = story, no_json)]
pub struct SelectLocale(SharedString);

actions!(llmupchat, [About, Quit, CloseWindow, OpenWebsite]);

pub fn init(cx: &mut App) {
    gpui_component::init(cx);
    themes::init(cx);
    cx.bind_keys([KeyBinding::new("cmd-q", Quit, None)]);

    cx.on_action(|_: &Quit, cx: &mut App| {
        cx.quit();
    });

    cx.activate(true);
}

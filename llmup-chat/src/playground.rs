use gpui::*;
use gpui_component::{
    ActiveTheme, blue_100, blue_200, blue_800, green_100, green_800,
    group_box::GroupBox,
    input::{InputState, TextInput},
    label::Label,
    resizable::{ResizableState, h_resizable, resizable_panel},
    sidebar::{Sidebar, SidebarHeader, SidebarMenu, SidebarMenuItem},
    v_flex,
};

use crate::helper::section;

pub struct Playground {
    collapsed: bool,
    input1: Entity<InputState>,
    history: Vec<Entity<History>>,
    sidebar_state: Entity<ResizableState>,
    search_input: Entity<InputState>,
}

/// An asset source that loads assets from the `./assets` folder.
pub struct History {
    pub title: String,
    #[allow(dead_code)]
    pub content: String,
}

impl Playground {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let hist = History {
            title: String::from("some chat"),
            content: String::from("this is the content of the chat"),
        };
        let hist = cx.new(|_| hist);
        let search_input = cx.new(|cx| InputState::new(window, cx).placeholder("Search..."));
        Self {
            collapsed: false,
            input1: cx.new(|cx| InputState::new(window, cx)),
            history: vec![hist],
            sidebar_state: ResizableState::new(cx),
            search_input,
        }
    }
}

pub(crate) fn user_input(is_dark: bool, content: impl Into<SharedString>) -> impl IntoElement {
    let green = if is_dark { green_800() } else { green_100() };

    //GroupBox::new().bg(green_100()).child(Label::new(content))
    GroupBox::new().border_4().child(
        div()
            .ml_20()
            .bg(green)
            .rounded_2xl()
            .text_align(TextAlign::Left)
            .child(Label::new(content).m_3().line_height(rems(1.8))),
    )
}

pub(crate) fn agent_message(is_dark: bool, content: impl Into<SharedString>) -> impl IntoElement {
    let blue = if is_dark { blue_800() } else { blue_100() };
    GroupBox::new().border_4().child(
        div()
            .bg(blue)
            .mr_20()
            .rounded_2xl()
            .text_align(TextAlign::Left)
            .child(Label::new(content).m_3().line_height(rems(1.8))),
    )
}

impl Render for Playground {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let _query = self.search_input.read(cx).value().trim().to_lowercase();

        let histories = self.history.iter().collect::<Vec<_>>();
        let is_dark = cx.theme().is_dark();

        h_resizable("llmchat-container", self.sidebar_state.clone())
            .child(
                v_flex()
                    .flex_1()
                    .h_full()
                    .overflow_x_hidden()
                    .child(
                        div()
                            .h_full()
                            .mx_1()
                            .my_10()
                            .child(user_input(is_dark, "Tell me a story about rust"))
                            .child(agent_message(
                                is_dark,
                                "This is a story about a vector that had elements that enter a bar",
                            )),
                    )
                    .child(
                        section("Input")
                            .max_w_md()
                            //.size(px(40.))
                            .child(TextInput::new(&self.input1).bg(blue_200())),
                    )
                    .into_any_element(),
            )
            .child(
                resizable_panel()
                    .size(px(255.))
                    .size_range(px(200.)..px(320.))
                    .child(
                        Sidebar::right()
                            .width(relative(1.))
                            .border_width(px(0.))
                            .collapsed(self.collapsed)
                            .header(
                                v_flex()
                                    .w_full()
                                    .gap_4()
                                    .child(
                                        SidebarHeader::new().w_full().child(Label::new("History")),
                                    )
                                    .child(
                                        div()
                                            .bg(cx.theme().sidebar_accent)
                                            .px_1()
                                            .rounded_full()
                                            .flex_1()
                                            .mx_1()
                                            .child(
                                                TextInput::new(&self.search_input)
                                                    .appearance(false)
                                                    .cleanable(),
                                            ),
                                    ),
                            )
                            .child(
                                SidebarMenu::new().children(
                                    histories.clone().into_iter().enumerate().map(
                                        |(_idx, history)| {
                                            //SidebarMenu::menu().children()
                                            SidebarMenuItem::new(history.read(cx).title.clone())
                                        },
                                    ),
                                ),
                            ),
                    ),
            )
    }
}
